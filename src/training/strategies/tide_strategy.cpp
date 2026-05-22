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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>

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
        // read from it during initialize() to seed the SOA scratch.
        lfs::core::SplatData* placeholder = nullptr;

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

        // Sizing. Phase 3.2b: take the placeholder's Gaussian count as the
        // working-set size. Phase 3.3 will switch to capacity_blocks *
        // block_size.
        const std::size_t n = impl_->placeholder->size();
        if (n == 0) {
            throw std::runtime_error("TideStrategy::initialize: placeholder has 0 Gaussians");
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

        LOG_DEBUG("TideStrategy: allocated {} bytes of SOA scratch for {} Gaussians (SH degree {}, shN floats/g {})",
                  impl_->scratch_bytes, n, impl_->sh_degree, impl_->shN_floats);

        // Bootstrap from placeholder so the view contains valid data.
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

        LOG_INFO("TideStrategy initialized (shell): {} Gaussians, SH degree {}, view-backed SplatData",
                 n, impl_->sh_degree);
    }

    // ----------------------------------------------------------------------

    void TideStrategy::pre_step(int /*iter*/, RenderOutput& /*render_output*/) {
        // Phase 3.2b: no-op. Phase 3.3 will call
        // working_set_->wait_and_activate() + aos_to_soa(d_*) here.
    }

    void TideStrategy::post_backward(int /*iter*/, RenderOutput& /*render_output*/) {
        // Phase 3.2b: no-op. Phase 3.3 will flag dirty blocks here so the
        // next prefetch knows what to write back.
    }

    void TideStrategy::step(int iter) {
        if (impl_->params == nullptr || impl_->optimizer == nullptr || impl_->scheduler == nullptr) {
            LOG_ERROR("TideStrategy::step called before initialize");
            return;
        }
        if (iter >= static_cast<int>(impl_->params->iterations)) {
            return;
        }
        // Phase 3.2b: equivalent to MCMC::step's optimizer dance. Phase 3.3
        // will replace .step(iter) with TideResidentAdam::step over the SOA
        // scratch and kick off WorkingSet::prefetch(next_blocks).
        impl_->optimizer->step(iter);
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

} // namespace lfs::training
