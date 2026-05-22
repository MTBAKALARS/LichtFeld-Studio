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
#include <cstdint>
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
        void prefetch_next(int next_iter, const lfs::core::Camera& next_cam) override;
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

        // Phase 3.5.4 test accessors --------------------------------------
        /// True iff the most recent `pre_forward` exercised the LRU+visibility
        /// eviction branch (moments enabled AND num_blocks > capacity). False
        /// for the iota-all-blocks fast path (legacy v1 stores or v2 stores
        /// that fit fully in WorkingSet capacity).
        bool last_pre_forward_used_lru() const noexcept;
        /// Per-block last-used iteration counter. Returns -1 if the block has
        /// never been resident (i.e. never selected in any `pre_forward`).
        /// In-process only — no checkpoint persistence (future phase).
        std::int64_t block_last_used_iter(std::size_t block_id) const noexcept;
        /// Snapshot of the resident block_id list from the most recent
        /// `pre_forward` (caller copies; vector reference is invalidated on
        /// the next `pre_forward`). For the LRU path this is the visible set
        /// plus the LRU-fill blocks, sorted ascending by block_id.
        const std::vector<std::size_t>& last_resident_block_ids() const noexcept;

        // Phase 3.5.6 pipelined-prefetch test accessors --------------------
        /// True iff a prefetch has been issued by `prefetch_next` but not yet
        /// consumed by `pre_forward` (i.e. WorkingSet::prefetch_pending()).
        bool has_pending_prefetch() const noexcept;
        /// Snapshot of the predicted resident block_id list staged by the
        /// most recent `prefetch_next` call. Cleared when `pre_forward`
        /// consumes (or drains) it.
        const std::vector<std::size_t>& last_prefetched_ids() const noexcept;
        /// Aggregate counters for the pipelined-prefetch state machine.
        /// Monotonically non-decreasing across the lifetime of the strategy.
        struct PrefetchStats {
            std::uint64_t prefetches_issued          = 0; ///< prefetch_next that actually issued a WS::prefetch.
            std::uint64_t prefetch_hits              = 0; ///< pre_forward consumed via wait_and_activate AND ids matched.
            std::uint64_t prefetch_misses            = 0; ///< pre_forward drained a stale prefetch then sync-loaded.
            std::uint64_t sync_loads                 = 0; ///< pre_forward fell through to WS::load_and_activate.
            std::uint64_t prefetch_skipped_no_change = 0; ///< prefetch_next short-circuited because predicted == last_loaded_ids.
        };
        PrefetchStats prefetch_stats() const noexcept;

    private:
        // Phase 3.5.6: shared core of `pre_forward` and `prefetch_next`. Pure
        // computation — writes the visible block_id list to `out_visible`, the
        // predicted resident block_id list to `out_resident`, and the LRU-branch
        // flag to `out_used_lru`. May throw `std::runtime_error` when Mode C
        // visible set exceeds capacity. Uses `impl_->bounds_scratch` and
        // `planes_scratch` as internal temporaries; does NOT touch
        // `last_visible_ids`, `last_pre_forward_used_lru`, `block_last_used_iter`,
        // `visible_scratch`, or `resident_scratch`.
        void compute_resident_set_(int iter,
                                   const lfs::core::Camera& cam,
                                   std::vector<std::size_t>& out_visible,
                                   std::vector<std::size_t>& out_resident,
                                   bool& out_used_lru);

        // Phase 3.5.7 fix: unpack active WorkingSet AOS blocks into the d_*
        // SOA scratch buffers that back the rendering SplatData view. Must
        // be called BEFORE `fast_rasterize_forward` reads `get_model()`,
        // because trainer.cpp calls `pre_step` (the original unpack site)
        // AFTER render. Without this hook the view-backed SplatData starts
        // zero-filled at init (see initialize() bootstrap-skip when WS is
        // attached) and the first render returns 0 visible primitives,
        // causing trainer.cpp:2459 to early-return forever (tiles_processed
        // == 0). Called from pre_forward after every load_and_activate /
        // wait_and_activate that brings new data into the WS. Idempotent
        // and cheap (one kernel launch per active block).
        void unpack_active_data_to_soa_();

        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::training
