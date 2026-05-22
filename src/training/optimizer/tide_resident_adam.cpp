/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file tide_resident_adam.cpp
 * @brief Phase 3.1 Adam optimizer for the Tide working-set path.
 *
 * Owns per-(ParamType) exp_avg / exp_avg_sq device buffers; accepts param/grad
 * as external raw device pointers. Reuses @c fast_lfs::optimizer::adam_step_raw
 * verbatim so updates are numerically identical to @ref AdamOptimizer.
 *
 * Phase 3.1 does NOT integrate with @ref WorkingSet (no per-block routing,
 * no BlockStore writeback). It is a self-contained optimizer surface plus the
 * minimum bookkeeping the trainer will eventually rely on.
 */

#include "optimizer/tide_resident_adam.hpp"

#include "adam_api.h"  // fast_lfs::optimizer::adam_step_raw
#include "core/logger.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstring>
#include <string>

namespace lfs::training::tide {

    namespace {

        std::string cuda_err(std::string_view what, cudaError_t e) {
            return std::string(what) + ": " + cudaGetErrorString(e);
        }

        constexpr std::size_t kNumParamTypes = AdamOptimizer::all_param_types().size();

        std::size_t type_index(ParamType type) noexcept {
            return static_cast<std::size_t>(type);
        }

        std::string_view type_name(ParamType type) noexcept {
            switch (type) {
            case ParamType::Means:    return "means";
            case ParamType::Sh0:      return "sh0";
            case ParamType::ShN:      return "shN";
            case ParamType::Scaling:  return "scaling";
            case ParamType::Rotation: return "rotation";
            case ParamType::Opacity:  return "opacity";
            }
            return "unknown";
        }

    } // namespace

    struct TideResidentAdam::Impl {
        struct ParamState {
            float* exp_avg = nullptr;      ///< Device buffer, num_elements floats
            float* exp_avg_sq = nullptr;   ///< Device buffer, num_elements floats
            std::size_t num_elements = 0;
            int64_t step_count = 0;
            bool allocated = false;
        };

        // Fixed-size table indexed by ParamType enum value.
        std::array<ParamState, kNumParamTypes> params{};
    };

    TideResidentAdam::~TideResidentAdam() {
        if (!impl_) return;
        for (auto& p : impl_->params) {
            if (p.exp_avg != nullptr) {
                cudaFree(p.exp_avg);
                p.exp_avg = nullptr;
            }
            if (p.exp_avg_sq != nullptr) {
                cudaFree(p.exp_avg_sq);
                p.exp_avg_sq = nullptr;
            }
        }
    }

    TideResidentAdam::TideResidentAdam(TideResidentAdam&&) noexcept = default;
    TideResidentAdam& TideResidentAdam::operator=(TideResidentAdam&&) noexcept = default;

    // ============================================================
    // create
    // ============================================================
    std::expected<std::unique_ptr<TideResidentAdam>, std::string>
    TideResidentAdam::create(const Config& config, std::span<const ParamSpec> params) {
        if (params.empty()) {
            return std::unexpected<std::string>(
                "TideResidentAdam::create: params must not be empty");
        }
        if (config.sh_warmup_iterations < 0) {
            return std::unexpected<std::string>(
                "TideResidentAdam::create: sh_warmup_iterations must be >= 0");
        }

        if (auto e = cudaSetDevice(config.cuda_device); e != cudaSuccess) {
            return std::unexpected(cuda_err("cudaSetDevice", e));
        }

        auto self = std::unique_ptr<TideResidentAdam>(new TideResidentAdam{});
        self->config_ = config;
        self->impl_ = std::make_unique<Impl>();

        // Track which ParamTypes we've seen so we can reject duplicates.
        std::array<bool, kNumParamTypes> seen{};
        seen.fill(false);

        for (const auto& spec : params) {
            const std::size_t idx = type_index(spec.type);
            if (idx >= kNumParamTypes) {
                return std::unexpected<std::string>(
                    "TideResidentAdam::create: invalid ParamType enum value");
            }
            if (seen[idx]) {
                return std::unexpected<std::string>(
                    std::string("TideResidentAdam::create: duplicate ParamType: ") +
                    std::string(type_name(spec.type)));
            }
            seen[idx] = true;

            auto& state = self->impl_->params[idx];
            state.num_elements = spec.num_elements;
            state.step_count = 0;
            state.allocated = true;

            if (spec.num_elements == 0) {
                // Allowed (e.g. ShN at SH degree 0). No allocation needed.
                continue;
            }

            const std::size_t bytes = spec.num_elements * sizeof(float);
            if (auto e = cudaMalloc(reinterpret_cast<void**>(&state.exp_avg), bytes);
                e != cudaSuccess) {
                return std::unexpected(cuda_err(
                    "TideResidentAdam::create: cudaMalloc exp_avg", e));
            }
            if (auto e = cudaMalloc(reinterpret_cast<void**>(&state.exp_avg_sq), bytes);
                e != cudaSuccess) {
                return std::unexpected(cuda_err(
                    "TideResidentAdam::create: cudaMalloc exp_avg_sq", e));
            }
            if (auto e = cudaMemset(state.exp_avg, 0, bytes); e != cudaSuccess) {
                return std::unexpected(cuda_err(
                    "TideResidentAdam::create: cudaMemset exp_avg", e));
            }
            if (auto e = cudaMemset(state.exp_avg_sq, 0, bytes); e != cudaSuccess) {
                return std::unexpected(cuda_err(
                    "TideResidentAdam::create: cudaMemset exp_avg_sq", e));
            }
        }

        LOG_DEBUG("TideResidentAdam: created with {} param types", params.size());
        return self;
    }

    // ============================================================
    // step
    // ============================================================
    std::expected<void, std::string>
    TideResidentAdam::step(ParamType type, float* param, const float* grad,
                           std::size_t num_elements, int iteration) {
        const std::size_t idx = type_index(type);
        if (idx >= kNumParamTypes || !impl_->params[idx].allocated) {
            return std::unexpected<std::string>(
                std::string("TideResidentAdam::step: param type not registered: ") +
                std::string(type_name(type)));
        }

        auto& state = impl_->params[idx];

        if (num_elements != state.num_elements) {
            return std::unexpected<std::string>(
                std::string("TideResidentAdam::step: num_elements mismatch for ") +
                std::string(type_name(type)) +
                " (got " + std::to_string(num_elements) +
                ", expected " + std::to_string(state.num_elements) + ")");
        }

        // Increment step_count BEFORE the warmup gate so that ShN's bookkeeping
        // stays consistent with the legacy AdamOptimizer (which also increments
        // before the early-return — adam_optimizer.cpp:180).
        state.step_count++;

        if (type == ParamType::ShN && iteration <= config_.sh_warmup_iterations) {
            stats_.steps_skipped++;
            return {};
        }

        if (num_elements == 0) {
            // Nothing to do (e.g. ShN with feature_dim 0). Still count as a step.
            stats_.steps_skipped++;
            return {};
        }

        if (param == nullptr || grad == nullptr) {
            return std::unexpected<std::string>(
                "TideResidentAdam::step: param/grad pointer is null");
        }

        // Per-iteration scalar bias correction terms. Match adam_optimizer.cpp:187-188.
        const double bias_correction1_rcp =
            1.0 / (1.0 - std::pow(config_.adam.beta1, state.step_count));
        const double bias_correction2_sqrt_rcp =
            1.0 / std::sqrt(1.0 - std::pow(config_.adam.beta2, state.step_count));

        // Per-param learning rate (falls back to base lr if not overridden).
        const auto& lrs = config_.adam.param_lrs;
        const auto lr_key = std::string(type_name(type));
        const float param_lr = lrs.contains(lr_key)
            ? static_cast<float>(lrs.at(lr_key))
            : config_.adam.lr;

        fast_lfs::optimizer::adam_step_raw(
            param,
            state.exp_avg,
            state.exp_avg_sq,
            grad,
            static_cast<int>(num_elements),
            param_lr,
            static_cast<float>(config_.adam.beta1),
            static_cast<float>(config_.adam.beta2),
            static_cast<float>(config_.adam.eps),
            static_cast<float>(bias_correction1_rcp),
            static_cast<float>(bias_correction2_sqrt_rcp));

        stats_.steps_total++;
        return {};
    }

    // ============================================================
    // reset
    // ============================================================
    std::expected<void, std::string>
    TideResidentAdam::reset(ParamType type) {
        const std::size_t idx = type_index(type);
        if (idx >= kNumParamTypes || !impl_->params[idx].allocated) {
            return std::unexpected<std::string>(
                std::string("TideResidentAdam::reset: param type not registered: ") +
                std::string(type_name(type)));
        }
        auto& state = impl_->params[idx];
        state.step_count = 0;
        if (state.num_elements == 0) {
            return {};
        }
        const std::size_t bytes = state.num_elements * sizeof(float);
        if (auto e = cudaMemset(state.exp_avg, 0, bytes); e != cudaSuccess) {
            return std::unexpected(cuda_err(
                "TideResidentAdam::reset: cudaMemset exp_avg", e));
        }
        if (auto e = cudaMemset(state.exp_avg_sq, 0, bytes); e != cudaSuccess) {
            return std::unexpected(cuda_err(
                "TideResidentAdam::reset: cudaMemset exp_avg_sq", e));
        }
        return {};
    }

    // ============================================================
    // Accessors
    // ============================================================
    bool TideResidentAdam::has_param(ParamType type) const noexcept {
        const std::size_t idx = type_index(type);
        return idx < kNumParamTypes && impl_ && impl_->params[idx].allocated;
    }

    int64_t TideResidentAdam::step_count(ParamType type) const noexcept {
        const std::size_t idx = type_index(type);
        if (idx >= kNumParamTypes || !impl_) return 0;
        return impl_->params[idx].step_count;
    }

    const float* TideResidentAdam::exp_avg(ParamType type) const noexcept {
        const std::size_t idx = type_index(type);
        if (idx >= kNumParamTypes || !impl_) return nullptr;
        return impl_->params[idx].exp_avg;
    }

    const float* TideResidentAdam::exp_avg_sq(ParamType type) const noexcept {
        const std::size_t idx = type_index(type);
        if (idx >= kNumParamTypes || !impl_) return nullptr;
        return impl_->params[idx].exp_avg_sq;
    }

} // namespace lfs::training::tide
