/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file tide_runtime.hpp
 * @brief Phase 3.3b — trainer-side glue that owns the BlockStore + TieredCache +
 *        WorkingSet for a TideStrategy and wires them together.
 *
 * Lifetime: the runtime must outlive the strategy. Trainer holds it as a member
 * (typically `std::optional<TideRuntime>`) and destroys it AFTER the strategy
 * has been destructed, so the strategy's `WorkingSet` shared_ptr is still
 * valid through final `step` / serialization.
 */

#include "core/block_store.hpp"
#include "core/parameters.hpp"
#include "core/tiered_cache.hpp"
#include "strategies/tide_strategy.hpp"
#include "tide/working_set.hpp"

#include <expected>
#include <memory>
#include <string>

namespace lfs::training::tide {

    /// Owning bundle for a Tide training session.
    struct TideRuntime {
        std::shared_ptr<lfs::core::BlockStore> store;
        std::unique_ptr<lfs::core::TieredCache> cache;
        std::shared_ptr<WorkingSet> working_set;

        /// Effective WS capacity in blocks (after clamping to store size).
        std::size_t effective_capacity_blocks = 0;
        /// Total number of blocks in the underlying store.
        std::size_t store_num_blocks = 0;
    };

    /**
     * @brief Open the BlockStore at `opt.tide_store_path`, build TieredCache +
     *        WorkingSet sized from `opt`, and attach the WorkingSet to
     *        @p strategy via @ref TideStrategy::set_working_set.
     *
     * Must be called BEFORE @ref TideStrategy::initialize so the strategy's
     * SOA scratch can be sized from the WorkingSet config.
     *
     * Sizing rules:
     *  - effective WS capacity = `opt.tide_capacity_blocks` if non-zero,
     *    otherwise `store->num_blocks()`. Clamped to `store->num_blocks()`.
     *  - effective cache capacity = `opt.tide_cache_capacity_blocks` if
     *    non-zero, otherwise `2 * effective WS capacity` (min 4).
     *
     * @return populated TideRuntime on success, error string on failure.
     *         Returns an error if `opt.tide_store_path` is empty (caller is
     *         responsible for the "no tide store configured" early-return).
     */
    std::expected<TideRuntime, std::string> attach_tide_working_set(
        TideStrategy& strategy,
        const lfs::core::param::OptimizationParameters& opt);

    /**
     * @brief Activate ALL blocks of the runtime's store in the WorkingSet
     *        (single-tile mode). Equivalent to
     *        `ws.load_and_activate(cache, {0..N-1})`.
     *
     * Should be called AFTER `strategy.initialize` so the SOA scratch exists
     * but BEFORE the first `pre_step`. Phase 3.5 will replace this with
     * per-iteration frustum-driven prefetch.
     */
    std::expected<void, std::string> activate_all_blocks(TideRuntime& runtime);

} // namespace lfs::training::tide
