/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file tide_strategy.cpp
 * @brief Phase 3.2b — TideStrategy shell.
 *
 * Owns SOA scratch buffers (6 cudaMalloc allocations) and a view-backed
 * SplatData built with Tensor::from_blob over those buffers. The trainer
 * sees a normal SplatData / AdamOptimizer pair; the Tide working-set and
 * resident-Adam hookup land in Phase 3.3.
 */

#include "strategies/tide_strategy.hpp"

#include "core/block_store.hpp"
#include "core/camera.hpp"
#include "core/logger.hpp"
#include "core/tiered_cache.hpp"
#include "strategies/strategy_utils.hpp"
#include "tide/aos_soa_repack.hpp"
#include "tide/frustum_culler.hpp"
#include "tide/working_set.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <istream>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace lfs::training {

    namespace {

        constexpr std::uint32_t kTideMagic = 0x54494445u; // 'TIDE'
        constexpr std::uint32_t kTideVersion = 1u;

        std::string cuda_err(const char* what, cudaError_t err) {
            return std::string("TideStrategy: ") + what + " failed: " + cudaGetErrorString(err);
        }

        // Allocate `num_floats` floats on device 0. Throws std::runtime_error
        // on cudaMalloc failure so the IStrategy::initialize contract surfaces
        // a clean exception rather than a silent nullptr.
        float* cuda_alloc_floats(std::size_t num_floats) {
            if (num_floats == 0)
                return nullptr;
            float* ptr = nullptr;
            const auto err = cudaMalloc(reinterpret_cast<void**>(&ptr), num_floats * sizeof(float));
            if (err != cudaSuccess) {
                throw std::runtime_error(cuda_err("cudaMalloc(soa scratch)", err));
            }
            return ptr;
        }

        // shN is [N, sh_rest_components, 3]. For SH degree d:
        //   sh_rest_components = (d+1)^2 - 1, last dim 3 → flat per-Gaussian
        //   = ((d+1)^2 - 1) * 3.
        std::size_t shN_floats_per_gaussian(int sh_degree) {
            if (sh_degree <= 0)
                return 0;
            const std::size_t total = static_cast<std::size_t>(sh_degree + 1) * static_cast<std::size_t>(sh_degree + 1);
            return (total - 1) * 3;
        }

        // Phase 3.5.3e-2 helpers ------------------------------------------
        // Build SoaViews / MomentsSoaViews that point at the per-block slice
        // (gaussians [local_idx*g, (local_idx+1)*g) of the bulk SOA scratch,
        // for use with the existing aos_to_soa / soa_to_aos / moments_aos_to_soa
        // / moments_soa_to_aos kernels.
        tide::SoaViews per_block_data_views(float* d_means, float* d_scaling,
                                            float* d_rotation, float* d_opacity,
                                            float* d_sh0, float* d_shN,
                                            std::size_t local_idx, std::size_t g,
                                            std::size_t shN_floats) {
            const std::size_t off = local_idx * g;
            return tide::SoaViews{
                d_means + off * 3,
                d_scaling + off * 3,
                d_rotation + off * 4,
                d_opacity + off,
                d_sh0 + off * 3,
                d_shN == nullptr ? nullptr : d_shN + off * shN_floats,
                g, shN_floats};
        }

        tide::MomentsSoaViews per_block_moments_views(
            float* d_m_means, float* d_m_scaling, float* d_m_rotation,
            float* d_m_opacity, float* d_m_sh0, float* d_m_shN,
            float* d_v_means, float* d_v_scaling, float* d_v_rotation,
            float* d_v_opacity, float* d_v_sh0, float* d_v_shN,
            std::size_t local_idx, std::size_t g, std::size_t shN_floats) {
            const std::size_t off = local_idx * g;
            tide::SoaViews m{
                d_m_means + off * 3, d_m_scaling + off * 3, d_m_rotation + off * 4,
                d_m_opacity + off, d_m_sh0 + off * 3,
                d_m_shN == nullptr ? nullptr : d_m_shN + off * shN_floats,
                g, shN_floats};
            tide::SoaViews v{
                d_v_means + off * 3, d_v_scaling + off * 3, d_v_rotation + off * 4,
                d_v_opacity + off, d_v_sh0 + off * 3,
                d_v_shN == nullptr ? nullptr : d_v_shN + off * shN_floats,
                g, shN_floats};
            return tide::MomentsSoaViews{m, v};
        }

    } // namespace

    struct TideStrategy::Impl {
        // Bootstrap source — the SplatData handed to the constructor. We only
        // read from it during initialize() to seed the SOA scratch (if no
        // WorkingSet is attached) and to pick up sh_degree / scene_scale.
        lfs::core::SplatData* placeholder = nullptr;

        // Optional WorkingSet. When attached BEFORE initialize, SOA scratch is
        // sized to `capacity_blocks * gaussians_per_block` and pre_step / step
        // use the Tide-resident path. When null, the strategy degrades to the
        // Phase 3.2b shell (standard AdamOptimizer over the view-backed
        // SplatData).
        std::shared_ptr<tide::WorkingSet> working_set;

        // Cached optimization parameters (set in initialize / via
        // set_optimization_params).
        std::unique_ptr<const lfs::core::param::OptimizationParameters> params;

        // Working-set sizing. Phase 3.2b: equals the placeholder SplatData
        // size at initialize-time. Phase 3.3 will switch this to the
        // configured WorkingSet capacity (capacity_blocks * block_size).
        std::size_t soa_capacity = 0;
        int sh_degree = 0;
        std::size_t shN_floats = 0;

        // SOA scratch — six contiguous device allocations matching SplatData's
        // raw tensor layouts.
        float* d_means = nullptr;    // [N, 3]
        float* d_scaling = nullptr;  // [N, 3]
        float* d_rotation = nullptr; // [N, 4]
        float* d_opacity = nullptr;  // [N, 1]
        float* d_sh0 = nullptr;      // [N, 1, 3]
        float* d_shN = nullptr;      // [N, sh_rest_components, 3], may be null when sh_degree=0

        std::size_t scratch_bytes = 0;

        // View-backed SplatData. Tensors point at d_*; SplatData does not own
        // the buffers (Tensor::from_blob → non-owning views).
        std::unique_ptr<lfs::core::SplatData> splat_view;

        // Standard Adam over the view (for trainer compat — backward writes
        // grads here, regularization losses read state, etc.).
        std::unique_ptr<AdamOptimizer> optimizer;
        std::unique_ptr<ExponentialLR> scheduler;

        // Phase 3.1 resident Adam, allocated but not yet wired into step().
        // Phase 3.3 will route the .step() path through this instead of
        // `optimizer`.
        std::unique_ptr<tide::TideResidentAdam> resident_adam;

        // Phase 3.5.2 frustum-driven streaming sources.
        // BlockStore supplies per-block bounding spheres (snapshotted into
        // `bounds_scratch` once per pre_forward); TieredCache supplies the
        // pinned-host bytes for `WorkingSet::load_and_activate` on a miss.
        std::shared_ptr<lfs::core::BlockStore> store;
        lfs::core::TieredCache* cache = nullptr;

        // Camera-space near/far for frustum plane construction. Conservative
        // defaults — far is intentionally huge so distant scene blocks are
        // never near/far-culled by accident. Phase 3.5.4 will tighten these
        // from the scene scale.
        float frustum_near = 0.01f;
        float frustum_far  = 1.0e6f;

        // Reusable scratch buffers — avoid per-iter allocations.
        std::vector<lfs::core::BlockStore::BlockBounds> bounds_scratch;
        std::vector<std::size_t> visible_scratch;
        std::vector<std::size_t> resident_scratch; ///< Block IDs passed to load_and_activate.
        std::array<tide::FrustumCuller::Plane, 6> planes_scratch{};

        // Phase 3.5.2 telemetry / test surfaces.
        std::vector<std::size_t> last_visible_ids;
        bool last_pre_forward_loaded = false;
        // What `resident_scratch` looked like on the most recent
        // load_and_activate. Compared against the next `resident_scratch`
        // to skip a redundant load when the resident set is unchanged.
        std::vector<std::size_t> last_loaded_ids;

        // Phase 3.5.3e-2: per-block Adam moments SOA scratch. Only allocated
        // when the attached WorkingSet has `moments_bytes_per_block > 0` (v2
        // store). Mirrors the data SOA layout exactly, with TWO copies per
        // attribute (m and v) so step_external_moments can read/write moments
        // directly without an extra D2D pass. Sized to `soa_capacity`.
        float* d_m_means = nullptr;
        float* d_m_scaling = nullptr;
        float* d_m_rotation = nullptr;
        float* d_m_opacity = nullptr;
        float* d_m_sh0 = nullptr;
        float* d_m_shN = nullptr;
        float* d_v_means = nullptr;
        float* d_v_scaling = nullptr;
        float* d_v_rotation = nullptr;
        float* d_v_opacity = nullptr;
        float* d_v_sh0 = nullptr;
        float* d_v_shN = nullptr;
        std::size_t moments_scratch_bytes = 0;

        // Cached at initialize when a WorkingSet is attached:
        // bytes_per_block / kAosBytesPerGaussian. Drives per-slot offsets in
        // the v2 pre_step/step paths.
        std::size_t g_per_block = 0;

        // Phase 3.5.3e-2: per-block Adam step counter, keyed by global block_id.
        // In-process only — 3.5.7 will persist it next to the moments sidecar
        // so resume restores it. operator[]'s default zero-init is the
        // "fresh block" signal; the v2 step path increments to >=1 before
        // calling step_external_moments.
        std::unordered_map<std::size_t, std::int64_t> block_step_counts;

        // Telemetry: number of blocks the most recent v2 step iterated over.
        // Zero on the legacy v1 path (moments disabled).
        std::size_t last_step_block_count = 0;

        void free_scratch() noexcept {
            const std::array<float**, 6> ptrs{&d_means, &d_scaling, &d_rotation,
                                              &d_opacity, &d_sh0, &d_shN};
            for (auto* slot : ptrs) {
                if (*slot != nullptr) {
                    cudaFree(*slot);
                    *slot = nullptr;
                }
            }
            scratch_bytes = 0;
        }

        void free_moments_scratch() noexcept {
            const std::array<float**, 12> ptrs{
                &d_m_means, &d_m_scaling, &d_m_rotation, &d_m_opacity, &d_m_sh0, &d_m_shN,
                &d_v_means, &d_v_scaling, &d_v_rotation, &d_v_opacity, &d_v_sh0, &d_v_shN};
            for (auto* slot : ptrs) {
                if (*slot != nullptr) {
                    cudaFree(*slot);
                    *slot = nullptr;
                }
            }
            moments_scratch_bytes = 0;
        }
    };

    // ----------------------------------------------------------------------

    TideStrategy::TideStrategy(lfs::core::SplatData& placeholder_splat_data)
        : impl_(std::make_unique<Impl>()) {
        impl_->placeholder = &placeholder_splat_data;
    }

    TideStrategy::~TideStrategy() {
        if (!impl_)
            return;
        // Destruction order matters: scheduler holds optimizer ref, optimizer
        // holds splat_view ref, splat_view (Tensor views) reads d_*. Drop
        // them in reverse construction order before freeing scratch.
        impl_->resident_adam.reset();
        impl_->scheduler.reset();
        impl_->optimizer.reset();
        impl_->splat_view.reset();
        impl_->free_moments_scratch();
        impl_->free_scratch();
    }

    // ----------------------------------------------------------------------

    void TideStrategy::initialize(const lfs::core::param::OptimizationParameters& optimParams) {
        using namespace lfs::core;

        if (impl_->placeholder == nullptr) {
            throw std::runtime_error("TideStrategy::initialize: placeholder SplatData is null");
        }

        impl_->params = std::make_unique<const param::OptimizationParameters>(optimParams);

        // Sizing. If a WorkingSet is attached, size SOA scratch to its full
        // capacity (capacity_blocks * gaussians_per_block) so SOA buffers can
        // hold the largest possible resident set. Otherwise (Phase 3.2b shell)
        // size to the placeholder's Gaussian count.
        std::size_t n = 0;
        if (impl_->working_set) {
            const auto& cfg = impl_->working_set->config();
            if (cfg.bytes_per_block == 0 || cfg.bytes_per_block % tide::kAosBytesPerGaussian != 0) {
                throw std::runtime_error(
                    "TideStrategy::initialize: WorkingSet bytes_per_block (" +
                    std::to_string(cfg.bytes_per_block) + ") is not a multiple of " +
                    std::to_string(tide::kAosBytesPerGaussian));
            }
            const std::size_t g_per_block = cfg.bytes_per_block / tide::kAosBytesPerGaussian;
            n = cfg.capacity_blocks * g_per_block;
            if (n == 0) {
                throw std::runtime_error("TideStrategy::initialize: WorkingSet has 0 capacity");
            }
        } else {
            n = impl_->placeholder->size();
            if (n == 0) {
                throw std::runtime_error("TideStrategy::initialize: placeholder has 0 Gaussians");
            }
        }
        impl_->soa_capacity = n;
        impl_->sh_degree = impl_->placeholder->get_max_sh_degree();
        impl_->shN_floats = shN_floats_per_gaussian(impl_->sh_degree);

        // Allocate SOA scratch.
        impl_->d_means = cuda_alloc_floats(n * 3);
        impl_->d_scaling = cuda_alloc_floats(n * 3);
        impl_->d_rotation = cuda_alloc_floats(n * 4);
        impl_->d_opacity = cuda_alloc_floats(n * 1);
        impl_->d_sh0 = cuda_alloc_floats(n * 3);
        impl_->d_shN = (impl_->shN_floats == 0) ? nullptr : cuda_alloc_floats(n * impl_->shN_floats);
        impl_->scratch_bytes = sizeof(float) * n *
                               (3 + 3 + 4 + 1 + 3 + impl_->shN_floats);

        LOG_DEBUG("TideStrategy: allocated {} bytes of SOA scratch for {} Gaussians (SH degree {}, shN floats/g {}, working_set={})",
                  impl_->scratch_bytes, n, impl_->sh_degree, impl_->shN_floats,
                  impl_->working_set ? "attached" : "none");

        // Phase 3.5.3e-2: allocate per-block Adam moments SOA scratch when
        // the WorkingSet carries moments (v2 stores). When absent, the
        // strategy stays on the Phase 3.4c bulk path with TideResidentAdam's
        // internal m/v buffers and one shared step counter.
        const bool moments_enabled = impl_->working_set &&
                                     impl_->working_set->moments_bytes_per_block() > 0;
        if (moments_enabled) {
            impl_->g_per_block = impl_->working_set->config().bytes_per_block /
                                 tide::kAosBytesPerGaussian;
            // Sanity: each block's moments region must match the kernel's
            // expected per-Gaussian moments stride (118 floats).
            const std::size_t expected_mbpb =
                impl_->g_per_block * tide::kAosMomentsBytesPerGaussian;
            if (impl_->working_set->moments_bytes_per_block() != expected_mbpb) {
                throw std::runtime_error(
                    "TideStrategy::initialize: WorkingSet moments_bytes_per_block (" +
                    std::to_string(impl_->working_set->moments_bytes_per_block()) +
                    ") does not equal g_per_block * kAosMomentsBytesPerGaussian (" +
                    std::to_string(expected_mbpb) + ")");
            }

            auto alloc_pair = [n](float** dst_m, float** dst_v, std::size_t per_g) {
                const std::size_t count = n * per_g;
                if (count == 0) return;
                *dst_m = cuda_alloc_floats(count);
                *dst_v = cuda_alloc_floats(count);
                const auto e1 = cudaMemset(*dst_m, 0, count * sizeof(float));
                if (e1 != cudaSuccess) {
                    throw std::runtime_error(cuda_err("cudaMemset(m scratch)", e1));
                }
                const auto e2 = cudaMemset(*dst_v, 0, count * sizeof(float));
                if (e2 != cudaSuccess) {
                    throw std::runtime_error(cuda_err("cudaMemset(v scratch)", e2));
                }
            };
            alloc_pair(&impl_->d_m_means,    &impl_->d_v_means,    3);
            alloc_pair(&impl_->d_m_scaling,  &impl_->d_v_scaling,  3);
            alloc_pair(&impl_->d_m_rotation, &impl_->d_v_rotation, 4);
            alloc_pair(&impl_->d_m_opacity,  &impl_->d_v_opacity,  1);
            alloc_pair(&impl_->d_m_sh0,      &impl_->d_v_sh0,      3);
            if (impl_->shN_floats != 0) {
                alloc_pair(&impl_->d_m_shN, &impl_->d_v_shN, impl_->shN_floats);
            }
            impl_->moments_scratch_bytes = 2u * impl_->scratch_bytes;

            LOG_INFO("TideStrategy: allocated {} bytes of per-block Adam moments SOA scratch "
                     "(12 buffers, g_per_block={}, mirrors data SOA layout)",
                     impl_->moments_scratch_bytes, impl_->g_per_block);
        }

        // Bootstrap. When a WorkingSet is attached the SOA scratch will be
        // populated by pre_step()'s aos_to_soa() on the first iteration, so a
        // device-side zero-fill is sufficient (avoids reading garbage if
        // someone inspects the SplatData before the first pre_step). Otherwise
        // copy the placeholder's data so the view is immediately valid.
        if (impl_->working_set) {
            const std::array<std::pair<float*, std::size_t>, 6> zero_targets{{
                {impl_->d_means, n * 3},
                {impl_->d_scaling, n * 3},
                {impl_->d_rotation, n * 4},
                {impl_->d_opacity, n * 1},
                {impl_->d_sh0, n * 3},
                {impl_->d_shN, n * impl_->shN_floats},
            }};
            for (const auto& [ptr, count] : zero_targets) {
                if (ptr == nullptr || count == 0) continue;
                const auto err = cudaMemset(ptr, 0, count * sizeof(float));
                if (err != cudaSuccess) {
                    throw std::runtime_error(cuda_err("cudaMemset(soa zero)", err));
                }
            }
        } else {
            auto copy_from_placeholder = [n](float* dst, const Tensor& src, std::size_t expected_floats, const char* name) {
                if (!src.is_valid() || src.numel() == 0) {
                    if (expected_floats == 0)
                        return;
                    throw std::runtime_error(
                        std::string("TideStrategy::initialize: placeholder tensor '") +
                        name + "' is empty but expected " + std::to_string(expected_floats) + " floats");
                }
                if (src.numel() != expected_floats) {
                    throw std::runtime_error(
                        std::string("TideStrategy::initialize: placeholder tensor '") +
                        name + "' has " + std::to_string(src.numel()) +
                        " floats, expected " + std::to_string(expected_floats) +
                        " (n=" + std::to_string(n) + ")");
                }
                const auto err = cudaMemcpy(dst, src.ptr<float>(), expected_floats * sizeof(float),
                                            cudaMemcpyDeviceToDevice);
                if (err != cudaSuccess) {
                    throw std::runtime_error(cuda_err("cudaMemcpy(bootstrap)", err));
                }
            };

            copy_from_placeholder(impl_->d_means, impl_->placeholder->means_raw(), n * 3, "means");
            copy_from_placeholder(impl_->d_scaling, impl_->placeholder->scaling_raw(), n * 3, "scaling");
            copy_from_placeholder(impl_->d_rotation, impl_->placeholder->rotation_raw(), n * 4, "rotation");
            copy_from_placeholder(impl_->d_opacity, impl_->placeholder->opacity_raw(), n * 1, "opacity");
            copy_from_placeholder(impl_->d_sh0, impl_->placeholder->sh0_raw(), n * 3, "sh0");
            if (impl_->d_shN != nullptr) {
                copy_from_placeholder(impl_->d_shN, impl_->placeholder->shN_raw(), n * impl_->shN_floats, "shN");
            }
        }

        // Build the view-backed SplatData. from_blob returns a non-owning
        // Tensor that aliases the scratch pointer.
        const std::size_t sh_rest_components = (impl_->shN_floats == 0)
                                                   ? 0u
                                                   : (impl_->shN_floats / 3);
        Tensor means_view = Tensor::from_blob(impl_->d_means,
                                              TensorShape({n, 3}),
                                              Device::CUDA, DataType::Float32);
        Tensor scaling_view = Tensor::from_blob(impl_->d_scaling,
                                                TensorShape({n, 3}),
                                                Device::CUDA, DataType::Float32);
        Tensor rotation_view = Tensor::from_blob(impl_->d_rotation,
                                                 TensorShape({n, 4}),
                                                 Device::CUDA, DataType::Float32);
        Tensor opacity_view = Tensor::from_blob(impl_->d_opacity,
                                                TensorShape({n, 1}),
                                                Device::CUDA, DataType::Float32);
        Tensor sh0_view = Tensor::from_blob(impl_->d_sh0,
                                            TensorShape({n, 1, 3}),
                                            Device::CUDA, DataType::Float32);
        Tensor shN_view = (impl_->d_shN == nullptr)
                              ? Tensor::zeros({n, 0, 3}, Device::CUDA)
                              : Tensor::from_blob(impl_->d_shN,
                                                  TensorShape({n, sh_rest_components, 3}),
                                                  Device::CUDA, DataType::Float32);

        impl_->splat_view = std::make_unique<lfs::core::SplatData>(
            impl_->sh_degree,
            std::move(means_view),
            std::move(sh0_view),
            std::move(shN_view),
            std::move(scaling_view),
            std::move(rotation_view),
            std::move(opacity_view),
            impl_->placeholder->get_scene_scale());
        impl_->splat_view->set_active_sh_degree(impl_->placeholder->get_active_sh_degree());

        // Standard Adam over the view (Phase 3.2b: drives both grads and
        // state; Phase 3.3 will retain only the grad buffers and step via
        // TideResidentAdam).
        impl_->optimizer = create_optimizer(*impl_->splat_view, *impl_->params);
        impl_->optimizer->allocate_gradients(n);
        impl_->scheduler = create_scheduler(*impl_->params, *impl_->optimizer);

        // Phase 3.1 resident Adam, sharing the same per-ParamType element
        // counts. Reused config keeps LR/beta/eps numerically identical.
        const auto& adam_cfg = impl_->optimizer->get_config();
        tide::TideResidentAdam::Config tide_cfg;
        tide_cfg.adam = adam_cfg;
        tide_cfg.cuda_device = 0;
        tide_cfg.sh_warmup_iterations = 1000;

        const std::array<tide::TideResidentAdam::ParamSpec, 6> specs{{
            {ParamType::Means,     n * 3},
            {ParamType::Sh0,       n * 3},
            {ParamType::ShN,       n * impl_->shN_floats},
            {ParamType::Scaling,   n * 3},
            {ParamType::Rotation,  n * 4},
            {ParamType::Opacity,   n * 1},
        }};
        auto resident = tide::TideResidentAdam::create(
            tide_cfg, std::span<const tide::TideResidentAdam::ParamSpec>(specs));
        if (!resident.has_value()) {
            throw std::runtime_error("TideStrategy::initialize: TideResidentAdam::create failed: " + resident.error());
        }
        impl_->resident_adam = std::move(*resident);

        LOG_INFO("TideStrategy initialized: {} Gaussians, SH degree {}, view-backed SplatData, working_set={}",
                 n, impl_->sh_degree, impl_->working_set ? "attached" : "none");
    }

    // ----------------------------------------------------------------------

    void TideStrategy::pre_forward(int iter, const lfs::core::Camera& cam) {
        // Phase 3.5.2 — frustum-driven residency hook.
        //
        // Preconditions for the "real" path (all three must hold):
        //   - a WorkingSet is attached (set via set_working_set in attach_tide_working_set);
        //   - a BlockStore is attached (set via set_tide_sources, for bounds);
        //   - a TieredCache is attached (set via set_tide_sources, for byte source).
        // If any is missing the call is a no-op, matching the Phase 3.5.1
        // fallback behavior (resident set established elsewhere).
        if (!impl_->working_set || !impl_->store || !impl_->cache) {
            impl_->last_pre_forward_loaded = false;
            return;
        }

        // 1. Snapshot per-block bounds under one BlockStore lock acquisition.
        impl_->store->snapshot_bounds(impl_->bounds_scratch);
        const std::size_t num_blocks = impl_->bounds_scratch.size();
        if (num_blocks == 0) {
            // Empty store — nothing to do. Should not happen in practice
            // (open() rejects 0-block stores), but guard cleanly.
            impl_->last_visible_ids.clear();
            impl_->last_pre_forward_loaded = false;
            return;
        }

        // 2. Compute world-space frustum planes for this camera.
        tide::FrustumCuller::compute_frustum_planes(
            cam, impl_->frustum_near, impl_->frustum_far, impl_->planes_scratch);

        // 3. Cull all block bounding spheres against the 6 planes.
        tide::FrustumCuller::cull(
            std::span<const lfs::core::BlockStore::BlockBounds>(
                impl_->bounds_scratch.data(), impl_->bounds_scratch.size()),
            std::span<const tide::FrustumCuller::Plane, 6>(impl_->planes_scratch),
            impl_->visible_scratch);

        // 4. Publish telemetry. The visible list is what the camera ACTUALLY
        //    sees this iteration — this is the input that Phase 3.5.4 will
        //    forward to WorkingSet, once per-block Adam moments (Phase 3.5.3)
        //    have made variable residency safe.
        impl_->last_visible_ids = impl_->visible_scratch;

        // 5. Build the resident set we will hand to WorkingSet.
        //
        // *** Phase 3.5.2 design note (read carefully) ***
        // TideResidentAdam stores first/second-moment buffers as flat,
        // contiguous device arrays indexed by **position within the active
        // SOA**, not by block_id (Phase 3.1 deliberately landed the simplest
        // possible per-tile Adam). If we drove residency from the visible set
        // here, the active-SOA position of any given block_id would change
        // every iteration as cameras rotate, but the Adam moments would not
        // follow — Gaussian i's m/v from iter N would be applied to a
        // completely different Gaussian at iter N+1.
        //
        // Phase 3.5.3 (Adam moments stored per-block, paper Sec. 3.4) is the
        // prerequisite that makes variable residency numerically safe. Until
        // that lands, we deliberately force the resident set to include
        // every block in the store. The frustum cull result above is computed
        // (and validated by tests) but not yet acted on.
        //
        // This is intentional, not lazy: it preserves Phase 3.4c's exact
        // training trajectory while moving the activation call site from
        // trainer init into pre_forward, which is the architectural change
        // 3.5.2 is responsible for.
        impl_->resident_scratch.resize(num_blocks);
        std::iota(impl_->resident_scratch.begin(), impl_->resident_scratch.end(), std::size_t{0});

        // 6. Skip the load entirely when the resident set is identical to
        //    last time. For the all-blocks case this fires exactly once per
        //    training run (iter 0), giving v24 perf parity with Phase 3.4c.
        impl_->last_pre_forward_loaded = false;
        if (impl_->resident_scratch == impl_->last_loaded_ids) {
            return;
        }

        // 7. Synchronous load. Phase 3.5.6 will swap this for a pipelined
        //    prefetch (iter N+1's set staged while iter N computes), which is
        //    what unlocks SSD-bound 1B training. For 3.5.2 a blocking load
        //    matches what activate_all_blocks used to do.
        auto result = impl_->working_set->load_and_activate(
            *impl_->cache,
            std::span<const std::size_t>(impl_->resident_scratch.data(),
                                         impl_->resident_scratch.size()));
        if (!result) {
            throw std::runtime_error(
                "TideStrategy::pre_forward: WorkingSet::load_and_activate failed at iter " +
                std::to_string(iter) + ": " + result.error());
        }
        impl_->last_loaded_ids = impl_->resident_scratch;
        impl_->last_pre_forward_loaded = true;
    }

    void TideStrategy::pre_step(int /*iter*/, RenderOutput& /*render_output*/) {
        if (!impl_->working_set) {
            // Phase 3.2b shell path: no working set, SplatData view was
            // bootstrapped from the placeholder in initialize().
            return;
        }

        // Phase 3.3a: unpack the active AOS device buffer into SOA scratch.
        // Active gaussian count may be < soa_capacity if the active set is
        // partial; aos_to_soa only touches the first `num_gaussians` records.
        const std::size_t active_n = impl_->working_set->active_gaussian_count();
        if (active_n == 0) {
            LOG_WARN("TideStrategy::pre_step: working set has 0 active Gaussians; skipping unpack");
            return;
        }
        if (active_n > impl_->soa_capacity) {
            LOG_ERROR("TideStrategy::pre_step: active Gaussian count {} exceeds SOA capacity {}; clamping",
                      active_n, impl_->soa_capacity);
        }
        const std::size_t n = std::min(active_n, impl_->soa_capacity);

        const bool moments_enabled = impl_->working_set->moments_bytes_per_block() > 0;

        if (!moments_enabled) {
            // Legacy v1 bulk path: slot_bytes == bytes_per_block, so the
            // active buffer is a contiguous AOS of `active_n` Gaussians.
            const auto* aos = static_cast<const float*>(impl_->working_set->device_buffer());
            if (aos == nullptr) {
                throw std::runtime_error("TideStrategy::pre_step: WorkingSet has no active device buffer");
            }
            tide::SoaViews views{
                impl_->d_means, impl_->d_scaling, impl_->d_rotation,
                impl_->d_opacity, impl_->d_sh0, impl_->d_shN,
                n, impl_->shN_floats};
            const int rc = tide::aos_to_soa(aos, views, /*stream=*/nullptr);
            if (rc != 0) {
                throw std::runtime_error(
                    "TideStrategy::pre_step: aos_to_soa launch failed: " +
                    std::string(cudaGetErrorString(static_cast<cudaError_t>(rc))));
            }
            return;
        }

        // Phase 3.5.3e-2 v2 path: per-slot iteration. The per-slot stride is
        // `bytes_per_block + moments_bytes_per_block`, so a single bulk
        // aos_to_soa would mis-read the interleaved moments bytes as data.
        // We launch one aos_to_soa + one moments_aos_to_soa per active slot,
        // scattering into the matching per-block range of the bulk SOA / m,v
        // moments SOA scratch.
        const auto* base = static_cast<const std::byte*>(impl_->working_set->device_buffer());
        if (base == nullptr) {
            throw std::runtime_error("TideStrategy::pre_step: WorkingSet has no active device buffer");
        }
        const std::size_t slot_bytes = impl_->working_set->slot_bytes();
        const std::size_t g = impl_->g_per_block;
        const auto slices = impl_->working_set->active_slices();
        for (const auto& s : slices) {
            const float* slot_data = reinterpret_cast<const float*>(base + s.local_index * slot_bytes);
            tide::SoaViews dview = per_block_data_views(
                impl_->d_means, impl_->d_scaling, impl_->d_rotation,
                impl_->d_opacity, impl_->d_sh0, impl_->d_shN,
                s.local_index, g, impl_->shN_floats);
            const int rc1 = tide::aos_to_soa(slot_data, dview, /*stream=*/nullptr);
            if (rc1 != 0) {
                throw std::runtime_error(
                    "TideStrategy::pre_step: per-block aos_to_soa launch failed at slot " +
                    std::to_string(s.local_index) + ": " +
                    std::string(cudaGetErrorString(static_cast<cudaError_t>(rc1))));
            }

            const float* slot_moments = impl_->working_set->moments_device_buffer(s.local_index);
            if (slot_moments == nullptr) {
                throw std::runtime_error(
                    "TideStrategy::pre_step: missing moments buffer at slot " +
                    std::to_string(s.local_index));
            }
            tide::MomentsSoaViews mv = per_block_moments_views(
                impl_->d_m_means, impl_->d_m_scaling, impl_->d_m_rotation,
                impl_->d_m_opacity, impl_->d_m_sh0, impl_->d_m_shN,
                impl_->d_v_means, impl_->d_v_scaling, impl_->d_v_rotation,
                impl_->d_v_opacity, impl_->d_v_sh0, impl_->d_v_shN,
                s.local_index, g, impl_->shN_floats);
            const int rc2 = tide::moments_aos_to_soa(slot_moments, mv, /*stream=*/nullptr);
            if (rc2 != 0) {
                throw std::runtime_error(
                    "TideStrategy::pre_step: per-block moments_aos_to_soa launch failed at slot " +
                    std::to_string(s.local_index) + ": " +
                    std::string(cudaGetErrorString(static_cast<cudaError_t>(rc2))));
            }
        }
    }

    void TideStrategy::post_backward(int /*iter*/, RenderOutput& /*render_output*/) {
        // Phase 3.3a: no-op. Phase 3.3b will flag dirty blocks here so the
        // next prefetch knows what to write back to TieredCache / BlockStore.
    }

    void TideStrategy::step(int iter) {
        if (impl_->params == nullptr || impl_->optimizer == nullptr || impl_->scheduler == nullptr) {
            LOG_ERROR("TideStrategy::step called before initialize");
            return;
        }
        if (iter >= static_cast<int>(impl_->params->iterations)) {
            return;
        }

        if (!impl_->working_set) {
            // Phase 3.2b shell path: drive the held AdamOptimizer directly.
            impl_->optimizer->step(iter);
            impl_->optimizer->zero_grad(iter);
            impl_->scheduler->step();
            return;
        }

        const std::size_t active_n = impl_->working_set->active_gaussian_count();
        if (active_n == 0) {
            LOG_WARN("TideStrategy::step: working set has 0 active Gaussians; skipping");
            return;
        }
        const std::size_t n = std::min(active_n, impl_->soa_capacity);

        const bool moments_enabled = impl_->working_set->moments_bytes_per_block() > 0;
        impl_->last_step_block_count = 0;

        if (!moments_enabled) {
            // Phase 3.3a legacy path: TideResidentAdam owns m/v internally,
            // one shared per-type step counter, bulk SOA in/out.
            struct ParamRoute {
                ParamType type;
                float* param_ptr;
                std::size_t num_elements;
            };
            const std::array<ParamRoute, 6> routes{{
                {ParamType::Means,    impl_->d_means,    n * 3},
                {ParamType::Sh0,      impl_->d_sh0,      n * 3},
                {ParamType::ShN,      impl_->d_shN,      n * impl_->shN_floats},
                {ParamType::Scaling,  impl_->d_scaling,  n * 3},
                {ParamType::Rotation, impl_->d_rotation, n * 4},
                {ParamType::Opacity,  impl_->d_opacity,  n * 1},
            }};
            for (const auto& r : routes) {
                if (r.param_ptr == nullptr || r.num_elements == 0) continue;
                auto& grad_tensor = impl_->optimizer->get_grad(r.type);
                if (!grad_tensor.is_valid()) {
                    LOG_WARN("TideStrategy::step: grad tensor for ParamType {} is invalid; skipping", static_cast<int>(r.type));
                    continue;
                }
                const auto result = impl_->resident_adam->step(
                    r.type, r.param_ptr, grad_tensor.ptr<float>(),
                    r.num_elements, iter);
                if (!result.has_value()) {
                    throw std::runtime_error(
                        "TideStrategy::step: TideResidentAdam::step failed for ParamType " +
                        std::to_string(static_cast<int>(r.type)) + ": " + result.error());
                }
            }

            auto* aos = static_cast<float*>(impl_->working_set->mutable_device_buffer());
            if (aos == nullptr) {
                throw std::runtime_error("TideStrategy::step: WorkingSet has no mutable device buffer");
            }
            tide::SoaViews views{
                impl_->d_means, impl_->d_scaling, impl_->d_rotation,
                impl_->d_opacity, impl_->d_sh0, impl_->d_shN,
                n, impl_->shN_floats};
            const int rc = tide::soa_to_aos(views, aos, /*stream=*/nullptr);
            if (rc != 0) {
                throw std::runtime_error(
                    "TideStrategy::step: soa_to_aos launch failed: " +
                    std::string(cudaGetErrorString(static_cast<cudaError_t>(rc))));
            }

            impl_->optimizer->zero_grad(iter);
            impl_->scheduler->step();
            return;
        }

        // Phase 3.5.3e-2 v2 path: per-block Adam through step_external_moments.
        // The held TideResidentAdam's internal m/v are bypassed; each block's
        // m/v live in the WorkingSet moments slot (which travels with the
        // block on evict/re-admit, per Phase 3.5.3b round-trip), staged
        // through this strategy's per-attribute SOA m/v scratch.
        auto* base = static_cast<std::byte*>(impl_->working_set->mutable_device_buffer());
        if (base == nullptr) {
            throw std::runtime_error("TideStrategy::step: WorkingSet has no mutable device buffer");
        }
        const std::size_t slot_bytes = impl_->working_set->slot_bytes();
        const std::size_t g = impl_->g_per_block;
        const auto slices = impl_->working_set->active_slices();

        struct ParamRoute {
            ParamType type;
            float* param_base;
            float* m_base;
            float* v_base;
            std::size_t per_g;
        };
        const std::array<ParamRoute, 6> routes{{
            {ParamType::Means,    impl_->d_means,    impl_->d_m_means,    impl_->d_v_means,    3},
            {ParamType::Sh0,      impl_->d_sh0,      impl_->d_m_sh0,      impl_->d_v_sh0,      3},
            {ParamType::ShN,      impl_->d_shN,      impl_->d_m_shN,      impl_->d_v_shN,      impl_->shN_floats},
            {ParamType::Scaling,  impl_->d_scaling,  impl_->d_m_scaling,  impl_->d_v_scaling,  3},
            {ParamType::Rotation, impl_->d_rotation, impl_->d_m_rotation, impl_->d_v_rotation, 4},
            {ParamType::Opacity,  impl_->d_opacity,  impl_->d_m_opacity,  impl_->d_v_opacity,  1},
        }};

        for (const auto& s : slices) {
            const std::size_t local = s.local_index;
            const std::size_t off_g = local * g;

            // Bump per-block step counter (1-based, matches contract).
            auto& counter = impl_->block_step_counts[s.block_id];
            ++counter;
            const std::int64_t block_step = counter;

            for (const auto& r : routes) {
                if (r.param_base == nullptr || r.per_g == 0) continue;
                auto& grad_tensor = impl_->optimizer->get_grad(r.type);
                if (!grad_tensor.is_valid()) continue;
                float* param_slice = r.param_base + off_g * r.per_g;
                float* grad_slice  = grad_tensor.ptr<float>() + off_g * r.per_g;
                float* m_slice     = r.m_base + off_g * r.per_g;
                float* v_slice     = r.v_base + off_g * r.per_g;
                const std::size_t num = g * r.per_g;
                const auto result = impl_->resident_adam->step_external_moments(
                    r.type, param_slice, grad_slice, m_slice, v_slice,
                    num, block_step, iter);
                if (!result.has_value()) {
                    throw std::runtime_error(
                        "TideStrategy::step: step_external_moments failed at slot " +
                        std::to_string(local) + " (block " + std::to_string(s.block_id) +
                        ") for ParamType " + std::to_string(static_cast<int>(r.type)) +
                        ": " + result.error());
                }
            }

            // Repack this block's SOA → AOS (data + moments).
            float* slot_data = reinterpret_cast<float*>(base + local * slot_bytes);
            tide::SoaViews dview = per_block_data_views(
                impl_->d_means, impl_->d_scaling, impl_->d_rotation,
                impl_->d_opacity, impl_->d_sh0, impl_->d_shN,
                local, g, impl_->shN_floats);
            const int rc1 = tide::soa_to_aos(dview, slot_data, /*stream=*/nullptr);
            if (rc1 != 0) {
                throw std::runtime_error(
                    "TideStrategy::step: per-block soa_to_aos failed at slot " +
                    std::to_string(local) + ": " +
                    std::string(cudaGetErrorString(static_cast<cudaError_t>(rc1))));
            }

            float* slot_moments = impl_->working_set->moments_device_buffer(local);
            if (slot_moments == nullptr) {
                throw std::runtime_error(
                    "TideStrategy::step: missing moments buffer at slot " + std::to_string(local));
            }
            tide::MomentsSoaViews mv = per_block_moments_views(
                impl_->d_m_means, impl_->d_m_scaling, impl_->d_m_rotation,
                impl_->d_m_opacity, impl_->d_m_sh0, impl_->d_m_shN,
                impl_->d_v_means, impl_->d_v_scaling, impl_->d_v_rotation,
                impl_->d_v_opacity, impl_->d_v_sh0, impl_->d_v_shN,
                local, g, impl_->shN_floats);
            const int rc2 = tide::moments_soa_to_aos(mv, slot_moments, /*stream=*/nullptr);
            if (rc2 != 0) {
                throw std::runtime_error(
                    "TideStrategy::step: per-block moments_soa_to_aos failed at slot " +
                    std::to_string(local) + ": " +
                    std::string(cudaGetErrorString(static_cast<cudaError_t>(rc2))));
            }

            // Both regions just got fresh device-side values; mark dirty so
            // the next eviction writes them back to TieredCache + BlockStore
            // (the round-trip primitive landed in Phase 3.5.3b).
            impl_->working_set->mark_dirty(local, /*data=*/true, /*moments=*/true);
        }

        impl_->last_step_block_count = slices.size();
        impl_->optimizer->zero_grad(iter);
        impl_->scheduler->step();
    }

    // ----------------------------------------------------------------------

    lfs::core::SplatData& TideStrategy::get_model() {
        if (impl_->splat_view == nullptr) {
            throw std::runtime_error("TideStrategy::get_model: called before initialize");
        }
        return *impl_->splat_view;
    }

    const lfs::core::SplatData& TideStrategy::get_model() const {
        if (impl_->splat_view == nullptr) {
            throw std::runtime_error("TideStrategy::get_model: called before initialize");
        }
        return *impl_->splat_view;
    }

    AdamOptimizer& TideStrategy::get_optimizer() {
        if (impl_->optimizer == nullptr) {
            throw std::runtime_error("TideStrategy::get_optimizer: called before initialize");
        }
        return *impl_->optimizer;
    }

    const AdamOptimizer& TideStrategy::get_optimizer() const {
        if (impl_->optimizer == nullptr) {
            throw std::runtime_error("TideStrategy::get_optimizer: called before initialize");
        }
        return *impl_->optimizer;
    }

    // ----------------------------------------------------------------------

    void TideStrategy::remove_gaussians(const lfs::core::Tensor& /*mask*/) {
        // Phase 3.2b: structural removal on a view-backed SplatData is
        // unsupported. A real implementation needs to mark blocks dirty in
        // the BlockStore. For the shell, log and ignore.
        LOG_WARN("TideStrategy::remove_gaussians: ignored in Phase 3.2b shell");
    }

    void TideStrategy::serialize(std::ostream& os) const {
        const std::uint32_t magic = kTideMagic;
        const std::uint32_t version = kTideVersion;
        os.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        os.write(reinterpret_cast<const char*>(&version), sizeof(version));
        const std::uint64_t cap = impl_->soa_capacity;
        os.write(reinterpret_cast<const char*>(&cap), sizeof(cap));
    }

    void TideStrategy::deserialize(std::istream& is) {
        std::uint32_t magic = 0;
        std::uint32_t version = 0;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (magic != kTideMagic) {
            throw std::runtime_error("TideStrategy::deserialize: bad magic");
        }
        if (version != kTideVersion) {
            throw std::runtime_error("TideStrategy::deserialize: unsupported version " + std::to_string(version));
        }
        std::uint64_t cap = 0;
        is.read(reinterpret_cast<char*>(&cap), sizeof(cap));
        // Cap is informational in Phase 3.2b; real state load follows in
        // Phase 3.4 when BlockStore checkpointing is wired in.
        (void)cap;
    }

    // ----------------------------------------------------------------------

    void TideStrategy::reserve_optimizer_capacity(size_t capacity) {
        if (impl_->optimizer != nullptr) {
            impl_->optimizer->reserve_capacity(capacity);
        }
    }

    void TideStrategy::set_optimization_params(const lfs::core::param::OptimizationParameters& params) {
        impl_->params = std::make_unique<const lfs::core::param::OptimizationParameters>(params);
    }

    // ----------------------------------------------------------------------

    std::size_t TideStrategy::soa_capacity_gaussians() const noexcept {
        return impl_ ? impl_->soa_capacity : 0u;
    }

    std::size_t TideStrategy::soa_scratch_bytes() const noexcept {
        return impl_ ? impl_->scratch_bytes : 0u;
    }

    const tide::TideResidentAdam* TideStrategy::get_resident_adam() const noexcept {
        return impl_ ? impl_->resident_adam.get() : nullptr;
    }

    tide::WorkingSet* TideStrategy::get_working_set() noexcept {
        return impl_ ? impl_->working_set.get() : nullptr;
    }

    std::size_t TideStrategy::soa_moments_scratch_bytes() const noexcept {
        return impl_ ? impl_->moments_scratch_bytes : 0u;
    }

    std::size_t TideStrategy::last_step_block_count() const noexcept {
        return impl_ ? impl_->last_step_block_count : 0u;
    }

    std::int64_t TideStrategy::block_step_count(std::size_t block_id) const noexcept {
        if (!impl_) return 0;
        const auto it = impl_->block_step_counts.find(block_id);
        return it == impl_->block_step_counts.end() ? std::int64_t{0} : it->second;
    }

    void TideStrategy::set_working_set(std::shared_ptr<tide::WorkingSet> working_set) {
        if (impl_->splat_view != nullptr) {
            // SOA scratch sizing is decided in initialize(). Attaching a
            // WorkingSet afterwards would mean the SOA capacity no longer
            // matches the WorkingSet's capacity; reject explicitly rather
            // than silently mis-sizing.
            throw std::runtime_error(
                "TideStrategy::set_working_set: must be called BEFORE initialize() "
                "(SOA scratch is already sized)");
        }
        impl_->working_set = std::move(working_set);
    }

    void TideStrategy::set_tide_sources(std::shared_ptr<lfs::core::BlockStore> store,
                                        lfs::core::TieredCache* cache) {
        // Unlike set_working_set, this is safe to call either before or after
        // initialize(): pre_forward checks the pointers on every call and
        // simply no-ops when either is missing. Allowing late attachment keeps
        // the trainer wire-up order flexible.
        impl_->store = std::move(store);
        impl_->cache = cache;
    }

    std::size_t TideStrategy::last_visible_block_count() const noexcept {
        return impl_ ? impl_->last_visible_ids.size() : 0u;
    }

    const std::vector<std::size_t>& TideStrategy::last_visible_block_ids() const noexcept {
        static const std::vector<std::size_t> empty;
        return impl_ ? impl_->last_visible_ids : empty;
    }

    bool TideStrategy::last_pre_forward_loaded() const noexcept {
        return impl_ ? impl_->last_pre_forward_loaded : false;
    }

} // namespace lfs::training
