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
#include "core/point_cloud.hpp"
#include "core/tiered_cache.hpp"
#include "io/exporter.hpp"
#include "io/formats/ply.hpp"
#include "strategies/strategy_utils.hpp"
#include "tide/aos_soa_repack.hpp"
#include "tide/frustum_culler.hpp"
#include "tide/working_set.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <expected>
#include <filesystem>
#include <istream>
#include <numeric>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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

        // Phase 3.5.4: per-block last-used iteration counter, keyed by global
        // block_id. Stamped to `iter` for every block that is made resident in
        // pre_forward (i.e. every visible block + every LRU-fill block). Used
        // when `|visible| < capacity < num_blocks` to pick which non-visible
        // blocks fill the remaining slots (highest last_used_iter wins).
        // In-process only; not persisted across restarts.
        std::unordered_map<std::size_t, std::int64_t> block_last_used_iter;

        // Phase 3.5.4 telemetry: did the most recent `pre_forward` exercise
        // the LRU eviction branch (moments_enabled && num_blocks > capacity)?
        // Equals false for the iota-all-blocks fast path.
        bool last_pre_forward_used_lru = false;

        // Phase 3.5.6 pipelined prefetch state.
        // `pending_prefetch_ids` mirrors the block_id list passed to
        // `WorkingSet::prefetch` by the most recent `prefetch_next` call. It
        // is cleared whenever pre_forward consumes (or drains) the staged
        // prefetch. `working_set->prefetch_pending()` is the source of truth
        // for whether an async copy is in flight; this vector is the predicted
        // set used to decide hit vs miss against `resident_scratch`.
        std::vector<std::size_t> pending_prefetch_ids;
        TideStrategy::PrefetchStats prefetch_stats{};

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

            // Phase 3.5.8c VRAM-FIX: release placeholder PLY's GPU tensors.
            // In the WorkingSet-attached path the SOA scratch is bootstrapped
            // by zero-fill (above), and pre_step's aos_to_soa pulls the real
            // data from the WorkingSet's per-slot buffers on first iteration.
            // The placeholder SplatData loaded from PLY (~1.66 GiB GPU for
            // 6.95M Gaussians SH-3) is dead weight from here on: TideStrategy
            // exposes its view-backed SplatData (from_blob over d_*) instead.
            // Freeing the placeholder tensors recovers the VRAM that the
            // rasterizer arena (512 MB) and per-step grad/scratch need.
            //
            // Phase 3.5.8w live-GUI-render bridge: instead of clearing the
            // placeholder tensors to invalid (which would leave the visualizer
            // staring at `_means.is_valid() == false` for the entire training
            // run), reassign each placeholder tensor to a NON-OWNING
            // `Tensor::from_blob` view of the matching d_* SOA scratch buffer.
            // The reassignment destructs the old owning PLY tensor in place
            // (freeing the ~1.66 GiB GPU memory as before), then alias-installs
            // a zero-cost handle that lets `Scene::getTrainingModel()` →
            // viewer renderer read the live training state. The shape is
            // initialized to `{0, ...}` so the renderer sees "no data yet"
            // until the first `unpack_active_data_to_soa_()` call publishes
            // an `active_n > 0` view via `update_render_view_`.
            if (impl_->placeholder != nullptr) {
                size_t free_before = 0, total_b = 0;
                cudaMemGetInfo(&free_before, &total_b);
                impl_->placeholder->means_raw()    = lfs::core::Tensor::from_blob(
                    impl_->d_means,    TensorShape({0, 3}), Device::CUDA, DataType::Float32);
                impl_->placeholder->scaling_raw()  = lfs::core::Tensor::from_blob(
                    impl_->d_scaling,  TensorShape({0, 3}), Device::CUDA, DataType::Float32);
                impl_->placeholder->rotation_raw() = lfs::core::Tensor::from_blob(
                    impl_->d_rotation, TensorShape({0, 4}), Device::CUDA, DataType::Float32);
                impl_->placeholder->opacity_raw()  = lfs::core::Tensor::from_blob(
                    impl_->d_opacity,  TensorShape({0, 1}), Device::CUDA, DataType::Float32);
                impl_->placeholder->sh0_raw()      = lfs::core::Tensor::from_blob(
                    impl_->d_sh0,      TensorShape({0, 1, 3}), Device::CUDA, DataType::Float32);
                // shN may be absent (sh_degree == 0); use an empty view-backed
                // tensor of the canonical shape so callers don't NPE on shape().
                const std::size_t sh_rest_components =
                    (impl_->shN_floats == 0) ? 0u : (impl_->shN_floats / 3);
                impl_->placeholder->shN_raw() = (impl_->d_shN == nullptr)
                    ? lfs::core::Tensor::zeros({0, 0, 3}, Device::CUDA)
                    : lfs::core::Tensor::from_blob(impl_->d_shN,
                          TensorShape({0, sh_rest_components, 3}),
                          Device::CUDA, DataType::Float32);
                cudaDeviceSynchronize();
                // Phase 3.5.8e: do NOT trim_cached_memory() here. Although
                // trim makes the freed bytes visible to cudaMemGetInfo, it
                // forces downstream allocations (gradient buffers, Adam
                // state in the optimizer) to cudaMalloc fresh which causes
                // fragmentation and ends up costing more VRAM than is saved.
                // The placeholder bytes go back to the size-bucketed pool
                // cache, where they are reused by subsequent identically-
                // sized requests. Net VRAM use is the same.
                size_t free_after = 0;
                cudaMemGetInfo(&free_after, &total_b);
                const long long delta_mib = (static_cast<long long>(free_after) -
                                             static_cast<long long>(free_before)) /
                                            (1024 * 1024);
                LOG_INFO("TideStrategy: released placeholder PLY GPU tensors "
                         "(pool-visible freed {} MiB; free {} -> {} MiB)",
                         delta_mib,
                         free_before / (1024u * 1024u),
                         free_after / (1024u * 1024u));
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
        //
        // Phase 3.5.8b VRAM fix: in v2 path (moments_enabled), the view-
        // backed AdamOptimizer's step() is never called — the trainer drives
        // step_external_moments on the per-block WorkingSet moments slot.
        // create_optimizer would pre-allocate m/v buffers sized for
        // params.max_cap (typ. 5M) — ~2.3 GB of dead VRAM. Override to 0 so
        // Adam state is allocated lazily on first step() (which never fires
        // in v2). Gradient buffers (allocate_gradients) are still allocated
        // because the rasterizer writes them and step_external_moments reads
        // them.
        if (moments_enabled) {
            auto opt_params_no_prealloc = *impl_->params;
            opt_params_no_prealloc.max_cap = 0;
            impl_->optimizer = create_optimizer(*impl_->splat_view, opt_params_no_prealloc);
            LOG_INFO("TideStrategy: view-backed AdamOptimizer state pre-allocation SKIPPED (v2 path bypasses optimizer->step); only gradient buffers allocated");
        } else {
            impl_->optimizer = create_optimizer(*impl_->splat_view, *impl_->params);
        }
        impl_->optimizer->allocate_gradients(n);
        impl_->scheduler = create_scheduler(*impl_->params, *impl_->optimizer);

        // Phase 3.1 resident Adam, sharing the same per-ParamType element
        // counts. Reused config keeps LR/beta/eps numerically identical.
        //
        // Phase 3.5.8b VRAM fix: in v2 path (moments_enabled), each block's
        // m/v live in the WorkingSet moments slot and step_external_moments
        // is called with EXTERNAL pointers — TideResidentAdam's internal
        // exp_avg/exp_avg_sq buffers are never read or written. Allocating
        // them costs ~3 GB for a 6.55M-Gaussian SH-3 model (ShN alone is
        // 2.36 GB), pushing us past the 24 GB physical limit. Pass
        // num_elements=0 to skip the cudaMalloc while keeping the ParamSpec
        // registered (state.allocated=true), which step_external_moments
        // requires. Legacy v1 path (no moments) keeps the original sizing.
        const auto& adam_cfg = impl_->optimizer->get_config();
        tide::TideResidentAdam::Config tide_cfg;
        tide_cfg.adam = adam_cfg;
        tide_cfg.cuda_device = 0;
        tide_cfg.sh_warmup_iterations = 1000;

        const std::size_t means_n    = moments_enabled ? 0u : n * 3;
        const std::size_t sh0_n      = moments_enabled ? 0u : n * 3;
        const std::size_t shN_n      = moments_enabled ? 0u : n * impl_->shN_floats;
        const std::size_t scaling_n  = moments_enabled ? 0u : n * 3;
        const std::size_t rotation_n = moments_enabled ? 0u : n * 4;
        const std::size_t opacity_n  = moments_enabled ? 0u : n * 1;

        const std::array<tide::TideResidentAdam::ParamSpec, 6> specs{{
            {ParamType::Means,     means_n},
            {ParamType::Sh0,       sh0_n},
            {ParamType::ShN,       shN_n},
            {ParamType::Scaling,   scaling_n},
            {ParamType::Rotation,  rotation_n},
            {ParamType::Opacity,   opacity_n},
        }};
        auto resident = tide::TideResidentAdam::create(
            tide_cfg, std::span<const tide::TideResidentAdam::ParamSpec>(specs));
        if (!resident.has_value()) {
            throw std::runtime_error("TideStrategy::initialize: TideResidentAdam::create failed: " + resident.error());
        }
        impl_->resident_adam = std::move(*resident);

        if (moments_enabled) {
            const std::size_t would_have_alloc_floats =
                n * (3u /*means*/ + 3u /*sh0*/ + impl_->shN_floats + 3u /*scaling*/ + 4u /*rotation*/ + 1u /*opacity*/);
            LOG_INFO("TideStrategy: TideResidentAdam internal m/v allocations SKIPPED (v2 path uses WorkingSet moments slot via step_external_moments); saved ~{} MiB",
                     (would_have_alloc_floats * 2u * sizeof(float)) / (1024u * 1024u));
        }

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

        // 1-5. Compute resident set (helper writes to visible_scratch +
        // resident_scratch). Phase 3.5.6 splits this out so `prefetch_next`
        // can reuse it with its own scratch buffers.
        bool used_lru = false;
        compute_resident_set_(iter, cam,
                              impl_->visible_scratch,
                              impl_->resident_scratch,
                              used_lru);

        // Publish telemetry from the computation step.
        impl_->last_visible_ids = impl_->visible_scratch;
        impl_->last_pre_forward_used_lru = used_lru;

        // Stamp last_used_iter for every block we are about to make resident.
        // This includes both visible blocks AND LRU-fill blocks; the latter
        // are "kept warm" by being resident even if not directly rendered.
        for (auto bid : impl_->resident_scratch) {
            impl_->block_last_used_iter[bid] = static_cast<std::int64_t>(iter);
        }

        impl_->last_pre_forward_loaded = false;

        // 6. Skip the load entirely when the resident set is identical to
        //    last time. For the all-blocks case this fires exactly once per
        //    training run (iter 0), giving v24 perf parity with Phase 3.4c.
        if (impl_->resident_scratch == impl_->last_loaded_ids) {
            // A pending prefetch (issued speculatively at end of last iter)
            // is unusable now; drain it so the WorkingSet state machine is
            // consistent for the next prefetch_next call.
            if (impl_->working_set->prefetch_pending()) {
                (void)impl_->working_set->wait_and_activate();
                impl_->pending_prefetch_ids.clear();
            }
            return;
        }

        // 7. Phase 3.5.6 pipelined consumption.
        //    If a prefetch is pending we MUST consume it via wait_and_activate
        //    (the WorkingSet contract forbids issuing a second prefetch before
        //    the first completes). If the staged ids match resident_scratch
        //    we are done; otherwise we drain and fall through to sync-load.
        if (impl_->working_set->prefetch_pending()) {
            const bool match = (impl_->pending_prefetch_ids == impl_->resident_scratch);
            auto wr = impl_->working_set->wait_and_activate();
            if (!wr) {
                throw std::runtime_error(
                    "TideStrategy::pre_forward: wait_and_activate failed at iter " +
                    std::to_string(iter) + ": " + wr.error());
            }
            const auto staged = std::move(impl_->pending_prefetch_ids);
            impl_->pending_prefetch_ids.clear();
            if (match) {
                impl_->last_loaded_ids = impl_->resident_scratch;
                impl_->last_pre_forward_loaded = true;
                ++impl_->prefetch_stats.prefetch_hits;
                // Phase 3.5.7 fix: populate d_* SOA scratch from the freshly
                // activated WS so the immediate render sees real Gaussians.
                unpack_active_data_to_soa_();
                return;
            }
            // Mismatch: GPU now holds `staged`, but we want `resident_scratch`.
            // Record what's actually on the device so the next equality-skip
            // check is honest, then sync-load the correct set below.
            impl_->last_loaded_ids = staged;
            ++impl_->prefetch_stats.prefetch_misses;
        }

        // 8. Synchronous load. Phase 3.5.6 retains this path as the fallback
        //    when no prefetch is pending or the prediction missed.
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
        ++impl_->prefetch_stats.sync_loads;
        // Phase 3.5.7 fix: populate d_* SOA scratch from the freshly loaded
        // WS so the immediate render sees real Gaussians (without this the
        // first render reads zero-initialized d_* and tiles_processed==0
        // triggers an early-return at trainer.cpp:2459).
        unpack_active_data_to_soa_();
    }

    void TideStrategy::prefetch_next(int next_iter, const lfs::core::Camera& next_cam) {
        // Phase 3.5.6 — speculative pipelined prefetch hook.
        // Soft-fail at every step: if anything goes wrong the next
        // `pre_forward` simply falls back to a synchronous load.
        if (!impl_->working_set || !impl_->store || !impl_->cache) {
            return;
        }
        // The WorkingSet contract forbids issuing a second prefetch before
        // the first completes. If a previous prefetch is still in flight we
        // skip this one; `pre_forward` will consume the in-flight one.
        if (impl_->working_set->prefetch_pending()) {
            return;
        }

        // Predict the resident set for (next_iter, next_cam) into LOCAL
        // buffers so we do not clobber `visible_scratch` / `resident_scratch`
        // / `last_loaded_ids` before `pre_forward` runs on the current iter.
        std::vector<std::size_t> predicted_visible;
        std::vector<std::size_t> predicted;
        bool used_lru = false;
        try {
            compute_resident_set_(next_iter, next_cam,
                                  predicted_visible, predicted, used_lru);
        } catch (const std::runtime_error&) {
            // Visible-set-exceeds-capacity or similar. Don't bring down the
            // training loop from here; `pre_forward` will throw the same
            // error on the matching call where the operator can see it.
            return;
        }

        // If `predicted == last_loaded_ids`, `pre_forward` will short-circuit
        // and skip the load anyway — issuing a prefetch would be wasted I/O.
        if (predicted == impl_->last_loaded_ids) {
            ++impl_->prefetch_stats.prefetch_skipped_no_change;
            return;
        }

        auto pr = impl_->working_set->prefetch(
            *impl_->cache,
            std::span<const std::size_t>(predicted.data(), predicted.size()));
        if (!pr) {
            // Soft-fail: leave pending_prefetch_ids empty; pre_forward will
            // see prefetch_pending() == false and sync-load.
            return;
        }
        impl_->pending_prefetch_ids = std::move(predicted);
        ++impl_->prefetch_stats.prefetches_issued;
    }

    void TideStrategy::compute_resident_set_(int iter,
                                             const lfs::core::Camera& cam,
                                             std::vector<std::size_t>& out_visible,
                                             std::vector<std::size_t>& out_resident,
                                             bool& out_used_lru) {
        // 1. Snapshot per-block bounds under one BlockStore lock acquisition.
        impl_->store->snapshot_bounds(impl_->bounds_scratch);
        const std::size_t num_blocks = impl_->bounds_scratch.size();
        out_used_lru = false;
        if (num_blocks == 0) {
            // Empty store — nothing to do. Should not happen in practice
            // (open() rejects 0-block stores), but guard cleanly.
            out_visible.clear();
            out_resident.clear();
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
            out_visible);

        // 4. Build the resident set.
        //
        // Three-mode policy (Phase 3.5.4):
        //   Mode A (v1, no moments): iota-all (variable residency unsafe).
        //   Mode B (v2, fits): iota-all (no eviction needed).
        //   Mode C (v2, over capacity): visible-first + LRU fill.
        const bool moments_enabled = impl_->working_set->moments_bytes_per_block() > 0;
        const std::size_t capacity = impl_->working_set->config().capacity_blocks;

        if (!moments_enabled || num_blocks <= capacity) {
            out_resident.resize(num_blocks);
            std::iota(out_resident.begin(), out_resident.end(), std::size_t{0});
            return;
        }

        // Mode C: visible-first + LRU fill.
        out_used_lru = true;
        out_resident.clear();
        out_resident.reserve(capacity);

        // Phase 3.5.8q VRAM-FIX: when visible > capacity, truncate the
        // visible list to capacity instead of throwing. The dropped blocks
        // are not rendered this iteration but they persist in the store and
        // will be selected by frustum culling in later iterations. This is
        // a graceful degradation that lets training proceed on tight VRAM
        // budgets where Tide capacity is set conservatively. Frustum culling
        // already preserves a natural priority ordering (closer / more
        // central blocks first in some sense) so the head of `out_visible`
        // is a reasonable choice.
        if (out_visible.size() > capacity) {
            LOG_WARN("TideStrategy::pre_forward: visible set size {} exceeds "
                     "WorkingSet capacity {} at iter {}; truncating to capacity "
                     "(dropped {} blocks this iteration). To eliminate these "
                     "drops, increase --tide-capacity-blocks.",
                     out_visible.size(), capacity, iter,
                     out_visible.size() - capacity);
            out_visible.resize(capacity);
        }

        std::unordered_set<std::size_t> visible_set(
            out_visible.begin(), out_visible.end());

        for (auto vid : out_visible) {
            out_resident.push_back(vid);
        }

        const std::size_t remaining = capacity - visible_set.size();
        if (remaining > 0 && num_blocks > visible_set.size()) {
            std::vector<std::pair<std::int64_t, std::size_t>> candidates;
            candidates.reserve(num_blocks - visible_set.size());
            for (std::size_t bid = 0; bid < num_blocks; ++bid) {
                if (visible_set.count(bid)) continue;
                auto it = impl_->block_last_used_iter.find(bid);
                const std::int64_t lui = (it == impl_->block_last_used_iter.end())
                                             ? std::int64_t{-1}
                                             : it->second;
                candidates.emplace_back(lui, bid);
            }
            const std::size_t k = std::min(remaining, candidates.size());
            if (k < candidates.size()) {
                std::nth_element(
                    candidates.begin(), candidates.begin() + k, candidates.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
            }
            for (std::size_t i = 0; i < k; ++i) {
                out_resident.push_back(candidates[i].second);
            }
        }

        std::sort(out_resident.begin(), out_resident.end());
    }

    void TideStrategy::update_render_view_(std::size_t active_n) {
        // Phase 3.5.8w live-GUI render bridge — see header doc on
        // update_render_view_ for the rationale. We reassign the placeholder
        // SplatData's tensors to non-owning views over the SOA scratch
        // buffers with shape `{active_n, ...}`. Per-frame this is just six
        // CPU-side Tensor handle swaps (a few `std::shared_ptr`-cheap
        // operations) — no GPU work, no allocations, no copies. The
        // visualizer's `hasRenderableGaussians` gate then sees
        // `_means.is_valid() == true && size() == active_n`, and the
        // rasterizer reads the live training state from the same d_*
        // buffers the trainer is updating.
        using namespace lfs::core;
        if (impl_->placeholder == nullptr) {
            return;
        }
        // Clamp to the SOA capacity to defend against caller bugs; the
        // unpack path already clamps via std::min(active_n, soa_capacity).
        const std::size_t n = std::min(active_n, impl_->soa_capacity);
        const std::size_t sh_rest_components =
            (impl_->shN_floats == 0) ? 0u : (impl_->shN_floats / 3);

        impl_->placeholder->means_raw() = Tensor::from_blob(
            impl_->d_means, TensorShape({n, 3}), Device::CUDA, DataType::Float32);
        impl_->placeholder->scaling_raw() = Tensor::from_blob(
            impl_->d_scaling, TensorShape({n, 3}), Device::CUDA, DataType::Float32);
        impl_->placeholder->rotation_raw() = Tensor::from_blob(
            impl_->d_rotation, TensorShape({n, 4}), Device::CUDA, DataType::Float32);
        impl_->placeholder->opacity_raw() = Tensor::from_blob(
            impl_->d_opacity, TensorShape({n, 1}), Device::CUDA, DataType::Float32);
        impl_->placeholder->sh0_raw() = Tensor::from_blob(
            impl_->d_sh0, TensorShape({n, 1, 3}), Device::CUDA, DataType::Float32);
        impl_->placeholder->shN_raw() = (impl_->d_shN == nullptr)
            ? Tensor::zeros({n, 0, 3}, Device::CUDA)
            : Tensor::from_blob(impl_->d_shN,
                                TensorShape({n, sh_rest_components, 3}),
                                Device::CUDA, DataType::Float32);
    }

    void TideStrategy::unpack_active_data_to_soa_() {
        // Phase 3.5.7 fix: data-only AOS→SOA unpack so the rendering
        // SplatData view sees real Gaussians before fast_rasterize_forward
        // runs. Mirrors the data path of pre_step() but skips the moments
        // unpack (moments are only needed before step()). See header for
        // why this is needed.
        if (!impl_->working_set) {
            return;
        }
        const std::size_t active_n = impl_->working_set->active_gaussian_count();
        if (active_n == 0) {
            return;
        }
        const std::size_t n = std::min(active_n, impl_->soa_capacity);
        const bool moments_enabled = impl_->working_set->moments_bytes_per_block() > 0;

        if (!moments_enabled) {
            // Legacy v1 bulk path: slot_bytes == bytes_per_block, contiguous AOS.
            const auto* aos = static_cast<const float*>(impl_->working_set->device_buffer());
            if (aos == nullptr) {
                throw std::runtime_error(
                    "TideStrategy::unpack_active_data_to_soa_: WorkingSet has no active device buffer");
            }
            tide::SoaViews views{
                impl_->d_means, impl_->d_scaling, impl_->d_rotation,
                impl_->d_opacity, impl_->d_sh0, impl_->d_shN,
                n, impl_->shN_floats};
            const int rc = tide::aos_to_soa(aos, views, /*stream=*/nullptr);
            if (rc != 0) {
                throw std::runtime_error(
                    "TideStrategy::unpack_active_data_to_soa_: aos_to_soa launch failed: " +
                    std::string(cudaGetErrorString(static_cast<cudaError_t>(rc))));
            }
            // Phase 3.5.8w: publish the freshly unpacked active range to the
            // viewer-facing placeholder so Scene::getTrainingModel() renders
            // the live training state.
            update_render_view_(n);
            return;
        }

        // v2 per-slot path: stride includes moments interleaving.
        const auto* base = static_cast<const std::byte*>(impl_->working_set->device_buffer());
        if (base == nullptr) {
            throw std::runtime_error(
                "TideStrategy::unpack_active_data_to_soa_: WorkingSet has no active device buffer");
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
            const int rc = tide::aos_to_soa(slot_data, dview, /*stream=*/nullptr);
            if (rc != 0) {
                throw std::runtime_error(
                    "TideStrategy::unpack_active_data_to_soa_: per-block aos_to_soa launch failed at slot " +
                    std::to_string(s.local_index) + ": " +
                    std::string(cudaGetErrorString(static_cast<cudaError_t>(rc))));
            }
        }
        // Phase 3.5.8w: publish the freshly unpacked active range to the
        // viewer-facing placeholder so Scene::getTrainingModel() renders the
        // live training state. NOTE: for Mode C (LRU eviction over capacity)
        // the active slots are sparse and `{active_n, ...}` is a contiguous
        // approximation — accurate for Mode A and Mode B (the only paths
        // exercised today) and a known limitation otherwise.
        update_render_view_(n);
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

    bool TideStrategy::last_pre_forward_used_lru() const noexcept {
        return impl_ ? impl_->last_pre_forward_used_lru : false;
    }

    std::int64_t TideStrategy::block_last_used_iter(std::size_t block_id) const noexcept {
        if (!impl_) return -1;
        const auto it = impl_->block_last_used_iter.find(block_id);
        return it == impl_->block_last_used_iter.end() ? std::int64_t{-1} : it->second;
    }

    const std::vector<std::size_t>& TideStrategy::last_resident_block_ids() const noexcept {
        static const std::vector<std::size_t> empty;
        return impl_ ? impl_->resident_scratch : empty;
    }

    bool TideStrategy::has_pending_prefetch() const noexcept {
        return impl_ && impl_->working_set && impl_->working_set->prefetch_pending();
    }

    const std::vector<std::size_t>& TideStrategy::last_prefetched_ids() const noexcept {
        static const std::vector<std::size_t> empty;
        return impl_ ? impl_->pending_prefetch_ids : empty;
    }

    TideStrategy::PrefetchStats TideStrategy::prefetch_stats() const noexcept {
        return impl_ ? impl_->prefetch_stats : PrefetchStats{};
    }

    // --- Phase 3.5.9 Plan A: full-BlockStore PLY export ----------------------
    //
    // The default Trainer::save_ply path calls lfs::io::save_ply(strategy_->get_model())
    // which on TideStrategy only sees the WorkingSet residency (typically 5-25%
    // of the model on a 24 GB GPU at 30M+SH-3). This override walks the entire
    // on-disk BlockStore, SoA-unpacks every block's latest revision into host
    // vectors with the standard 3DGS PLY attribute ordering, and emits a single
    // monolithic PLY containing every Gaussian in the model.
    //
    // Cost: ~248 bytes per Gaussian held in host RAM during the unpack pass
    // (means/scaling/rotation/opacity/sh0/shN). At 28.9M splats SH-3 this is
    // ~7.2 GiB host RAM peak — well within a 64 GB box. A future Plan A-2
    // could stream-write the PLY without materializing all attributes at once.
    //
    // The SH-rest ordering transposition: BlockStore stores rest as
    // [c0_R, c0_G, c0_B, c1_R, c1_G, c1_B, ...] (coefficient-major,
    // channel-inner), but the canonical 3DGS PLY layout (matching `process_sh`
    // in to_point_cloud) is [ch0_c0..ch0_c14, ch1_c0..ch1_c14, ch2_c0..ch2_c14]
    // (channel-major, coefficient-inner). We transpose during unpack so the
    // output PLY loads correctly in any standard viewer (Babylon / PlayCanvas /
    // splatviewer.com / SuperSplat).
    std::optional<std::expected<void, std::string>>
    TideStrategy::save_full_ply(const std::filesystem::path& output_path, bool binary) {
        if (!impl_ || !impl_->store) {
            // No BlockStore attached — let the trainer fall back to the
            // standard get_model()-based save path.
            return std::nullopt;
        }

        auto& store = *impl_->store;
        const std::size_t num_blocks = store.num_blocks();
        const std::size_t block_size = store.block_size();
        const std::size_t bytes_per_block = store.bytes_per_block();

        if (num_blocks == 0 || block_size == 0) {
            return std::expected<void, std::string>{};
        }

        const std::size_t total_gaussians = num_blocks * block_size;
        constexpr std::size_t kAosFloats = tide::kAosFloatsPerGaussian; // 59

        LOG_INFO("TideStrategy::save_full_ply: streaming {} blocks × {} = {} "
                 "Gaussians (SH-3) to {}",
                 num_blocks, block_size, total_gaussians,
                 output_path.string());

        const auto t_start = std::chrono::steady_clock::now();

        // Host-side flat attribute buffers (AoS-by-Gaussian, matching the
        // shapes expected by PointCloud → write_ply_binary).
        std::vector<float> host_means(total_gaussians * 3);
        std::vector<float> host_scaling(total_gaussians * 3);
        std::vector<float> host_rotation(total_gaussians * 4);
        std::vector<float> host_opacity(total_gaussians * 1);
        std::vector<float> host_sh0(total_gaussians * 3);
        std::vector<float> host_shN(total_gaussians * 45);

        std::vector<std::byte> block_buf(bytes_per_block);

        for (std::size_t b = 0; b < num_blocks; ++b) {
            auto read_result = store.read_block(b, std::span<std::byte>(block_buf));
            if (!read_result) {
                return std::unexpected(std::format(
                    "TideStrategy::save_full_ply: read_block({}) failed: {}",
                    b, read_result.error()));
            }

            const float* g_base = reinterpret_cast<const float*>(block_buf.data());

            for (std::size_t i = 0; i < block_size; ++i) {
                const std::size_t gid = b * block_size + i;
                const float* g = g_base + i * kAosFloats;

                // means [N, 3]
                host_means[gid * 3 + 0] = g[tide::kAosOffsetMeans + 0];
                host_means[gid * 3 + 1] = g[tide::kAosOffsetMeans + 1];
                host_means[gid * 3 + 2] = g[tide::kAosOffsetMeans + 2];

                // scaling [N, 3] (raw, pre-exp)
                host_scaling[gid * 3 + 0] = g[tide::kAosOffsetScaling + 0];
                host_scaling[gid * 3 + 1] = g[tide::kAosOffsetScaling + 1];
                host_scaling[gid * 3 + 2] = g[tide::kAosOffsetScaling + 2];

                // rotation [N, 4] (raw quaternion xyzw; viewer normalizes)
                host_rotation[gid * 4 + 0] = g[tide::kAosOffsetRotation + 0];
                host_rotation[gid * 4 + 1] = g[tide::kAosOffsetRotation + 1];
                host_rotation[gid * 4 + 2] = g[tide::kAosOffsetRotation + 2];
                host_rotation[gid * 4 + 3] = g[tide::kAosOffsetRotation + 3];

                // opacity [N, 1] (raw, pre-sigmoid)
                host_opacity[gid] = g[tide::kAosOffsetOpacity];

                // sh0 [N, 3] = DC (R, G, B) — same order in BlockStore and PLY
                host_sh0[gid * 3 + 0] = g[tide::kAosOffsetSh0 + 0];
                host_sh0[gid * 3 + 1] = g[tide::kAosOffsetSh0 + 1];
                host_sh0[gid * 3 + 2] = g[tide::kAosOffsetSh0 + 2];

                // shN [N, 45] — transpose coefficient-major (BlockStore) to
                // channel-major (PLY). BlockStore: k = c*3 + ch (15 coeffs ×
                // 3 channels). PLY: write_idx = ch*15 + c.
                for (std::size_t k = 0; k < tide::kAosShNMaxFloats; ++k) {
                    const std::size_t c = k / 3;
                    const std::size_t ch = k % 3;
                    const std::size_t write_idx = ch * 15 + c;
                    host_shN[gid * 45 + write_idx] = g[tide::kAosOffsetShN + k];
                }
            }
        }

        // Build a PointCloud with CPU tensors (no GPU upload — at 28.9M ×
        // 248 B = 7.2 GiB this would blow past the 24 GB VRAM budget).
        lfs::core::PointCloud pc;
        pc.means = lfs::core::Tensor::from_vector(host_means, {total_gaussians, 3}, lfs::core::Device::CPU);
        pc.normals = lfs::core::Tensor::zeros({total_gaussians, 3}, lfs::core::Device::CPU);
        pc.scaling = lfs::core::Tensor::from_vector(host_scaling, {total_gaussians, 3}, lfs::core::Device::CPU);
        pc.rotation = lfs::core::Tensor::from_vector(host_rotation, {total_gaussians, 4}, lfs::core::Device::CPU);
        pc.opacity = lfs::core::Tensor::from_vector(host_opacity, {total_gaussians, 1}, lfs::core::Device::CPU);
        // sh0 buffer is [R, G, B] per Gaussian → shape [N, 3, 1] (channel, coeff).
        // shN buffer is [ch*15 + c] per Gaussian → shape [N, 3, 15] (channel, coeff).
        // This matches PointCloud's documented shape convention and the layout
        // expected by write_ply_binary's f_rest_* attribute enumeration.
        pc.sh0 = lfs::core::Tensor::from_vector(host_sh0, {total_gaussians, 3, 1}, lfs::core::Device::CPU);
        pc.shN = lfs::core::Tensor::from_vector(host_shN, {total_gaussians, 3, 15}, lfs::core::Device::CPU);

        // 3DGS PLY attribute name order matches to_point_cloud/get_ply_attribute_names:
        // x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2 rot_0..3.
        pc.attribute_names = {"x", "y", "z", "nx", "ny", "nz"};
        pc.attribute_names.reserve(6 + 3 + 45 + 1 + 3 + 4);
        for (int i = 0; i < 3; ++i) pc.attribute_names.push_back("f_dc_" + std::to_string(i));
        for (int i = 0; i < 45; ++i) pc.attribute_names.push_back("f_rest_" + std::to_string(i));
        pc.attribute_names.push_back("opacity");
        for (int i = 0; i < 3; ++i) pc.attribute_names.push_back("scale_" + std::to_string(i));
        for (int i = 0; i < 4; ++i) pc.attribute_names.push_back("rot_" + std::to_string(i));

        const auto t_unpacked = std::chrono::steady_clock::now();
        const auto unpack_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_unpacked - t_start).count();
        LOG_INFO("TideStrategy::save_full_ply: unpacked {} Gaussians from {} blocks in {} ms — writing PLY",
                 total_gaussians, num_blocks, unpack_ms);

        // Force synchronous binary write — the async path would let our host
        // vectors go out of scope while the writer is still consuming them.
        // Plan A is correctness-first; throughput from-disk-to-disk on this
        // path is dominated by sequential read of the BlockStore anyway.
        const lfs::io::PlySaveOptions opts{
            .output_path = output_path,
            .binary = binary,
            .async = false};

        auto save_result = lfs::io::save_ply(pc, opts);
        if (!save_result) {
            return std::unexpected(std::format(
                "TideStrategy::save_full_ply: lfs::io::save_ply failed: {}",
                save_result.error().message));
        }

        const auto t_done = std::chrono::steady_clock::now();
        const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_done - t_start).count();
        LOG_INFO("TideStrategy::save_full_ply: wrote {} Gaussians to {} in {} ms total",
                 total_gaussians, output_path.string(), total_ms);

        return std::expected<void, std::string>{};
    }

} // namespace lfs::training
