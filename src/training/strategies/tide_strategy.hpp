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
#include "tide/working_set.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace lfs::core {
    class BlockStore;
    class TieredCache;
} // namespace lfs::core

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

        /// Attach a WorkingSet. Must be called BEFORE @ref initialize so that
        /// SOA scratch is sized to `capacity_blocks * gaussians_per_block`
        /// rather than the placeholder SplatData size. When attached:
        ///  - `pre_step` unpacks the active AOS device buffer into SOA scratch
        ///    via @ref tide::aos_to_soa.
        ///  - `step` invokes @ref tide::TideResidentAdam per ParamType and
        ///    repacks SOA back into the active buffer via @ref tide::soa_to_aos.
        /// When NOT attached the strategy falls back to the Phase 3.2b shell
        /// behavior (standard AdamOptimizer over the view-backed SplatData).
        void set_working_set(std::shared_ptr<tide::WorkingSet> working_set);

        /// Phase 3.5.2: hand the strategy the BlockStore (for per-iter bounds
        /// snapshots fed to the frustum culler) and the TieredCache (for the
        /// `WorkingSet::load_and_activate` source on a frustum-miss).
        ///
        /// Both pointers must outlive the strategy (TideRuntime owns them).
        /// When the strategy has a WorkingSet AND both sources, `pre_forward`
        /// performs full frustum-driven residency selection per iteration.
        /// When either source is null the strategy falls back to the Phase
        /// 3.5.1 behavior (no-op pre_forward, resident set established by an
        /// external `activate_all_blocks` call).
        void set_tide_sources(std::shared_ptr<lfs::core::BlockStore> store,
                              lfs::core::TieredCache* cache);

        // IStrategy interface ----------------------------------------------
        void initialize(const lfs::core::param::OptimizationParameters& optimParams) override;
        void pre_forward(int iter, const lfs::core::Camera& cam) override;
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
        /// Access the attached WorkingSet (may be null).
        tide::WorkingSet* get_working_set() noexcept;

        // Phase 3.5.3e-2 test accessors ------------------------------------
        /// Bytes allocated for per-block Adam moments SOA scratch (sum of
        /// 12 buffers: 6 m_* + 6 v_*). Zero when the attached WorkingSet
        /// has `moments_bytes_per_block() == 0` (legacy v1 path) or before
        /// `initialize()` runs.
        std::size_t soa_moments_scratch_bytes() const noexcept;
        /// Number of blocks the most recent `step()` iterated over. Zero on
        /// the legacy v1 path (moments disabled) or before any step.
        std::size_t last_step_block_count() const noexcept;
        /// Per-block Adam step counter for `block_id`. Returns 0 if the block
        /// has never been stepped. In-process only (no checkpoint persistence
        /// until Phase 3.5.7).
        std::int64_t block_step_count(std::size_t block_id) const noexcept;

        // Phase 3.5.2 test accessors --------------------------------------
        /// Number of blocks the most recent `pre_forward` call selected as
        /// visible. 0 before the first call, or when sources are missing.
        std::size_t last_visible_block_count() const noexcept;
        /// Snapshot of the visible block_id list from the most recent
        /// `pre_forward` (caller copies; vector reference is invalidated on
        /// the next `pre_forward`).
        const std::vector<std::size_t>& last_visible_block_ids() const noexcept;
        /// True iff the most recent `pre_forward` actually called
        /// `WorkingSet::load_and_activate` (vs. skipping because the visible
        /// set was unchanged). 0 before the first call.
        bool last_pre_forward_loaded() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::training
