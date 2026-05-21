/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/block_store.hpp"
#include "core/export.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <list>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace lfs::core {

    /**
     * @brief CPU-side LRU cache between a @ref BlockStore (SSD) and the GPU working set.
     *
     * Implements TideGS Sec. 3.4: warm cache between SSD and VRAM with a per-block
     * dirty bit and asynchronous write-back to patch segments. Two-step eviction
     * path is VRAM -> CPU -> SSD; only the CPU -> SSD step is implemented here
     * (the GPU side lives in `GPUWorkingSet`, Phase 2).
     *
     * Memory model:
     *   - A fixed-size pool of pinned host buffers is allocated up front, each one
     *     `BlockStore::bytes_per_block()` bytes. This is the only large allocation
     *     the cache ever performs — eviction reuses the buffer for the incoming block.
     *   - The LRU list is a std::list of node handles; the lookup map points at list iters.
     *   - Dirty blocks evicted from VRAM are inserted at the MRU end with @ref dirty=true
     *     and an async flusher thread writes them to SSD.
     *
     * Thread-safety:
     *   - @ref get / @ref pin_for_write / @ref unpin / @ref evict are safe under concurrent calls.
     *   - The async flusher runs on a single dedicated thread.
     */
    class LFS_CORE_API TieredCache {
    public:
        struct Config {
            std::size_t capacity_blocks = 0;          ///< Hard cap on resident block count
            std::size_t pinned_pool_size_blocks = 0;  ///< Pinned (page-locked) host buffers; 0 = same as capacity
            std::size_t flush_queue_high_watermark = 256;
        };

        struct Stats {
            std::atomic<std::uint64_t> hits{0};
            std::atomic<std::uint64_t> misses{0};
            std::atomic<std::uint64_t> evictions{0};
            std::atomic<std::uint64_t> async_flush_jobs{0};
            std::atomic<std::uint64_t> sync_flushes{0};
            std::atomic<std::uint64_t> dirty_blocks_resident{0};
        };

        TieredCache(std::shared_ptr<BlockStore> store, const Config& config);
        ~TieredCache();

        TieredCache(const TieredCache&) = delete;
        TieredCache& operator=(const TieredCache&) = delete;

        // === Read path ===

        /**
         * @brief Get a read-only view of a block. Pins the block until @ref unpin.
         *
         * On hit: returns immediately, moves the block to MRU.
         * On miss: synchronously reads from the underlying @ref BlockStore.
         *
         * The returned span is valid until the corresponding @ref unpin call.
         */
        std::expected<std::span<const std::byte>, std::string> get(std::size_t block_id);

        /**
         * @brief Get a writable view of a block. Pins and marks dirty on @ref unpin.
         *
         * Calling this commits the cache to write the block back to SSD when it's
         * eventually evicted (lazy write-back, TideGS Sec. 3.4).
         */
        std::expected<std::span<std::byte>, std::string> pin_for_write(std::size_t block_id);

        /**
         * @brief Release a pin previously acquired via @ref get or @ref pin_for_write.
         *
         * @param dirty If true, the block is marked dirty regardless of how it was acquired.
         */
        void unpin(std::size_t block_id, bool dirty);

        // === Eviction ===

        /**
         * @brief Drop @p k LRU clean blocks. Dirty blocks are skipped (caller can
         *        force a sync flush via @ref flush_dirty if needed).
         */
        std::size_t evict_clean(std::size_t k);

        /**
         * @brief Synchronously flush all dirty blocks to the underlying store.
         *
         * Used at consistency barriers (checkpoint, shutdown).
         */
        std::expected<void, std::string> flush_dirty();

        // === Introspection ===

        std::size_t resident_blocks() const noexcept;
        std::size_t capacity() const noexcept { return config_.capacity_blocks; }
        const Stats& stats() const noexcept { return *stats_; }

    private:
        struct Node {
            std::size_t block_id = 0;
            std::byte* buffer = nullptr;   ///< Points into pinned pool
            std::uint32_t pin_count = 0;
            bool dirty = false;
            // Iterator into lru_list_ for O(1) splice on access. Only valid while pinned == 0.
            // We could use std::list<Node>::iterator but circular type forces erased storage.
        };

        // Map block_id -> list iterator
        struct ListEntry {
            std::shared_ptr<Node> node;
        };
        using LruList = std::list<ListEntry>;

        std::shared_ptr<BlockStore> store_;
        Config config_{};
        std::size_t bytes_per_block_ = 0;

        // Pinned host memory pool. Allocated once; freed on destruction.
        std::byte* pinned_pool_ = nullptr;
        std::size_t pinned_pool_bytes_ = 0;
        std::vector<std::byte*> free_buffers_;

        // LRU bookkeeping. lru_list_ has the LRU at front, MRU at back.
        mutable std::mutex mutex_;
        LruList lru_list_;
        std::unordered_map<std::size_t, LruList::iterator> map_;

        // Async flush worker: producer/consumer queue of (block_id, buffer-snapshot).
        struct FlushJob {
            std::size_t block_id;
            std::vector<std::byte> payload; // Owned copy so the cache buffer can be reused immediately
        };
        std::mutex flush_mutex_;
        std::condition_variable flush_cv_;
        std::list<FlushJob> flush_queue_;
        std::atomic<bool> flush_stop_{false};
        std::thread flush_thread_;

        std::unique_ptr<Stats> stats_;

        // Helpers (called with mutex_ held unless noted).
        std::byte* allocate_buffer_();
        void release_buffer_(std::byte* buf);
        std::expected<LruList::iterator, std::string> insert_block_(std::size_t block_id);
        void touch_(LruList::iterator it);
        void evict_one_clean_();
        void enqueue_flush_(std::size_t block_id, std::byte* buffer);
        void flush_worker_();
    };

} // namespace lfs::core
