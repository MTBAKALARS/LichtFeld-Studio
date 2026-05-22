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

#include "core/logger.hpp"
#include "strategies/strategy_utils.hpp"
#include "tide/aos_soa_repack.hpp"
#include "tide/working_set.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
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

    void TideStrategy::pre_forward(int /*iter*/, const lfs::core::Camera& /*cam*/) {
        // Phase 3.5.1: callback wiring only. The trainer now invokes this hook
        // every iteration (just after camera selection, just before
        // rasterize_forward) regardless of whether a WorkingSet is attached.
        // Behavior is intentionally a no-op so this commit is a pure refactor:
        // the WorkingSet's resident set continues to be populated up-front by
        // trainer init via `activate_all_blocks`.
        //
        // Phase 3.5.2 will move the activation here: compute the camera's 6
        // frustum planes, run FrustumCuller against block bounds, and request
        // the visible block IDs from the WorkingSet (sync first, then async).
        // From that point on, the trainer-side `activate_all_blocks` call
        // becomes unreachable and is removed.
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

        // Phase 3.3a: route the actual Adam step through TideResidentAdam over
        // the SOA scratch, then repack SOA back into the active AOS buffer.
        // The held AdamOptimizer is only used as a grad-buffer host (its
        // `step()` is intentionally NOT called on this path).
        const std::size_t active_n = impl_->working_set->active_gaussian_count();
        if (active_n == 0) {
            LOG_WARN("TideStrategy::step: working set has 0 active Gaussians; skipping");
            return;
        }
        const std::size_t n = std::min(active_n, impl_->soa_capacity);

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

        // Repack SOA -> AOS so the resident bytes in the active buffer
        // reflect the post-Adam parameters. Phase 3.3b will mark these blocks
        // dirty so the next prefetch writes them back to TieredCache.
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

        // Clear grads + advance scheduler (matching MCMC::step semantics).
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

} // namespace lfs::training
