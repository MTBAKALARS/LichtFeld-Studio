/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tide/tide_runtime.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"

#include <algorithm>
#include <filesystem>
#include <numeric>
#include <vector>

namespace lfs::training::tide {

    namespace {
        constexpr std::size_t kDefaultCacheMultiplier = 2;
        constexpr std::size_t kMinCacheCapacity = 4;
    } // namespace

    std::expected<TideRuntime, std::string> attach_tide_working_set(
        TideStrategy& strategy,
        const lfs::core::param::OptimizationParameters& opt) {
        if (opt.tide_store_path.empty()) {
            return std::unexpected("attach_tide_working_set: tide_store_path is empty");
        }
        if (!std::filesystem::exists(opt.tide_store_path)) {
            return std::unexpected("attach_tide_working_set: tide_store_path does not exist: " +
                                   lfs::core::path_to_utf8(opt.tide_store_path));
        }

        // 1) Open the BlockStore.
        auto store_result = lfs::core::BlockStore::open(opt.tide_store_path);
        if (!store_result) {
            return std::unexpected("attach_tide_working_set: BlockStore::open failed: " + store_result.error());
        }
        std::shared_ptr<lfs::core::BlockStore> store(std::move(store_result.value()));

        const std::size_t n_blocks = store->num_blocks();
        if (n_blocks == 0) {
            return std::unexpected("attach_tide_working_set: store has 0 blocks");
        }

        // 2) Resolve effective capacities.
        std::size_t ws_capacity = (opt.tide_capacity_blocks == 0)
                                      ? n_blocks
                                      : opt.tide_capacity_blocks;
        if (ws_capacity > n_blocks) {
            LOG_WARN("attach_tide_working_set: tide_capacity_blocks={} > store.num_blocks={}; clamping",
                     ws_capacity, n_blocks);
            ws_capacity = n_blocks;
        }

        std::size_t cache_capacity = (opt.tide_cache_capacity_blocks == 0)
                                         ? std::max(kMinCacheCapacity, ws_capacity * kDefaultCacheMultiplier)
                                         : opt.tide_cache_capacity_blocks;
        if (cache_capacity < ws_capacity) {
            LOG_WARN("attach_tide_working_set: tide_cache_capacity_blocks={} < ws_capacity={}; bumping to ws_capacity",
                     cache_capacity, ws_capacity);
            cache_capacity = ws_capacity;
        }

        // 3) Build TieredCache.
        lfs::core::TieredCache::Config ccfg;
        ccfg.capacity_blocks = cache_capacity;
        auto cache = std::make_unique<lfs::core::TieredCache>(store, ccfg);

        // 4) Build WorkingSet sized from store->bytes_per_block.
        WorkingSet::Config wcfg;
        wcfg.capacity_blocks = ws_capacity;
        wcfg.bytes_per_block = store->bytes_per_block();
        wcfg.cuda_device = 0;
        auto ws_result = WorkingSet::create(wcfg);
        if (!ws_result) {
            return std::unexpected("attach_tide_working_set: WorkingSet::create failed: " + ws_result.error());
        }
        std::shared_ptr<WorkingSet> ws(std::move(ws_result.value()));

        // 5) Attach to strategy BEFORE initialize. May throw if initialize already ran.
        try {
            strategy.set_working_set(ws);
        } catch (const std::exception& e) {
            return std::unexpected(std::string("attach_tide_working_set: set_working_set failed: ") + e.what());
        }

        LOG_INFO("Tide runtime attached: store={} ({} blocks, {} B/block); WS capacity={}, cache capacity={}",
                 lfs::core::path_to_utf8(opt.tide_store_path),
                 n_blocks, store->bytes_per_block(), ws_capacity, cache_capacity);

        TideRuntime runtime;
        runtime.store = std::move(store);
        runtime.cache = std::move(cache);
        runtime.working_set = std::move(ws);
        runtime.effective_capacity_blocks = ws_capacity;
        runtime.store_num_blocks = n_blocks;
        return runtime;
    }

    std::expected<void, std::string> activate_all_blocks(TideRuntime& runtime) {
        if (!runtime.working_set || !runtime.cache) {
            return std::unexpected("activate_all_blocks: runtime is incomplete");
        }
        const std::size_t n = runtime.effective_capacity_blocks;
        if (n == 0) {
            return std::unexpected("activate_all_blocks: effective_capacity_blocks is 0");
        }
        std::vector<std::size_t> ids(n);
        std::iota(ids.begin(), ids.end(), 0u);
        auto r = runtime.working_set->load_and_activate(*runtime.cache, std::span<const std::size_t>(ids));
        if (!r) {
            return std::unexpected("activate_all_blocks: load_and_activate failed: " + r.error());
        }
        return {};
    }

} // namespace lfs::training::tide
