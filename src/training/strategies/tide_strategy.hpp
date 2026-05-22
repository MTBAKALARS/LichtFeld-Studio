/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "istrategy.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/scheduler.hpp"
#include "optimizer/tide_resident_adam.hpp"

#include <cstddef>
#include <memory>

namespace lfs::training {

    /**
     * @brief TideGS strategy — out-of-core 3DGS training via BlockStore +
     *        WorkingSet + tile-resident Adam.
     *
     * **Phase 3.2b — IStrategy SHELL only.** The strategy owns six SOA
     * scratch buffers in VRAM (means, scaling, rotation, opacity, sh0, shN)
     * and exposes them through a view-backed SplatData built with
     * `Tensor::from_blob`. On `initialize()` it bootstraps those buffers from
     * the placeholder SplatData passed at construction so the view contains
     * valid data and AdamOptimizer / TideResidentAdam can be constructed
     * around it.
     *
     * In this shell phase:
     *  - `pre_step` / `post_backward` are no-ops.
     *  - `step` advances the held AdamOptimizer + scheduler (functionally
     *    equivalent to MCMC's step for the resident set), so trainer
     *    integration is testable end-to-end.
     *  - The `TideResidentAdam` instance is registered against the same
     *    per-ParamType specs but is NOT yet routed into the step path.
     *
     * **Phase 3.3** will rewrite `pre_step` to call
     * `WorkingSet::wait_and_activate()` + `aos_to_soa()`, swap the
     * AdamOptimizer step for `TideResidentAdam::step()` operating directly
     * on the SOA scratch pointers, and kick off the next prefetch from
     * `step`.
     *
     * The constructor accepts a `SplatData&` for IStrategy compatibility,
     * but TideStrategy owns its own SplatData (the view-backed one) and
     * `get_model()` returns the owned instance, not the placeholder.
     */
    class TideStrategy : public IStrategy {
    public:
        TideStrategy() = delete;
        explicit TideStrategy(lfs::core::SplatData& placeholder_splat_data);
        ~TideStrategy() override;

        TideStrategy(const TideStrategy&) = delete;
        TideStrategy& operator=(const TideStrategy&) = delete;
        TideStrategy(TideStrategy&&) = delete;
        TideStrategy& operator=(TideStrategy&&) = delete;

        // IStrategy interface ----------------------------------------------
        void initialize(const lfs::core::param::OptimizationParameters& optimParams) override;
        void pre_step(int iter, RenderOutput& render_output) override;
        void post_backward(int iter, RenderOutput& render_output) override;
        void step(int iter) override;
        bool is_refining(int /*iter*/) const override { return false; }

        lfs::core::SplatData& get_model() override;
        const lfs::core::SplatData& get_model() const override;

        AdamOptimizer& get_optimizer() override;
        const AdamOptimizer& get_optimizer() const override;

        void remove_gaussians(const lfs::core::Tensor& mask) override;

        void serialize(std::ostream& os) const override;
        void deserialize(std::istream& is) override;
        const char* strategy_type() const override { return "tide"; }

        void reserve_optimizer_capacity(size_t capacity) override;
        void set_optimization_params(const lfs::core::param::OptimizationParameters& params) override;

        // Phase 3.2b test accessors ----------------------------------------
        /// SOA scratch capacity in Gaussians (== number of Gaussians in the
        /// view-backed SplatData after `initialize`).
        std::size_t soa_capacity_gaussians() const noexcept;
        /// Bytes allocated for SOA scratch (sum of all 6 buffers).
        std::size_t soa_scratch_bytes() const noexcept;
        /// Access the underlying Tide-resident Adam (testing only).
        const tide::TideResidentAdam* get_resident_adam() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::training
