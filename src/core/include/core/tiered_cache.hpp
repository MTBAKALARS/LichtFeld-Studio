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
            // Phase 3.5.5 — telemetry for the async writeback path.
            std::atomic<std::uint64_t> async_dirty_evictions{0};   ///< Dirty regions enqueued by eviction
            std::atomic<std::uint64_t> backpressure_waits{0};       ///< Times an enqueue blocked on watermark
            std::atomic<std::uint64_t> miss_drain_waits{0};         ///< Misses that had to wait for an in-flight async flush
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
         * @param dirty If true, the block's data region is marked dirty regardless
         *              of how it was acquired. The moments region's dirty bit is
         *              left alone (it tracks @ref pin_moments_for_write separately).
         */
        void unpin(std::size_t block_id, bool dirty);

        // === Adam moments (optional, present only if backing store has moments) ===

        /// True if the backing @ref BlockStore has an Adam moments sidecar region.
        bool has_moments() const noexcept { return moments_bytes_per_block_ != 0; }

        /// Per-block byte stride of the moments region (0 if absent).
        std::size_t moments_bytes_per_block() const noexcept { return moments_bytes_per_block_; }

        /**
         * @brief Read-only view of a block's Adam moments region.
         *
         * Caller MUST already hold a pin on @p block_id via @ref get or
         * @ref pin_for_write — this call does NOT take an additional pin.
         * The returned span lives in the same cache slot as the data view and is
         * valid until that pin is released.
         *
         * Errors if the cache has no moments region or the block is not pinned.
         */
        std::expected<std::span<const std::byte>, std::string> get_moments(std::size_t block_id);

        /**
         * @brief Writable view of a block's Adam moments. Marks the moments region
         *        dirty; subsequent @ref flush_dirty (or eviction with dirty moments)
         *        persists it via @ref BlockStore::write_moments.
         *
         * Caller MUST already hold a pin on @p block_id; this call does NOT take
         * an additional pin.
         */
        std::expected<std::span<std::byte>, std::string> pin_moments_for_write(std::size_t block_id);

        // === Eviction ===

        /**
         * @brief Drop @p k LRU clean blocks. Dirty blocks are skipped (caller can
         *        force a sync flush via @ref flush_dirty if needed).
         */
        std::size_t evict_clean(std::size_t k);

        /**
         * @brief Synchronously flush all dirty resident blocks AND drain any
         *        pending async write-backs to the underlying @ref BlockStore.
         *
         * Used at consistency barriers (checkpoint, shutdown). After this call
         * returns, every byte the caller wrote via @ref pin_for_write or
         * @ref pin_moments_for_write is durable on the underlying store, whether
         * it was still resident or had already been evicted with an in-flight
         * async flush in progress.
         */
        std::expected<void, std::string> flush_dirty();

        /**
         * @brief Block until the async flusher queue is empty and the worker is
         *        idle. Does NOT initiate any new flushes (use @ref flush_dirty
         *        for that). Returned the last error reported by the worker, if
         *        any, then clears it.
         *
         * Phase 3.5.5 API. Safe to call concurrently with @ref get / @ref unpin /
         * @ref evict_clean.
         */
        std::expected<void, std::string> drain_async_flushes();

        // === Introspection ===

        std::size_t resident_blocks() const noexcept;
        std::size_t capacity() const noexcept { return config_.capacity_blocks; }
        const Stats& stats() const noexcept { return *stats_; }

    private:
        struct Node {
            std::size_t block_id = 0;
            std::byte* buffer = nullptr;   ///< Points to start of slot in pinned pool
                                           ///< Layout: [data : bytes_per_block_] [moments : moments_bytes_per_block_]
            std::uint32_t pin_count = 0;
            bool dirty = false;            ///< Data region dirty
            bool moments_dirty = false;    ///< Moments region dirty (separate so we
                                           ///< don't write data when only moments changed)
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
        std::size_t bytes_per_block_ = 0;          ///< Data stride (BlockStore::bytes_per_block())
        std::size_t moments_bytes_per_block_ = 0;  ///< Moments stride (0 if store has no moments)
        std::size_t slot_bytes_ = 0;               ///< = bytes_per_block_ + moments_bytes_per_block_

        // Pinned host memory pool. Allocated once; freed on destruction.
        std::byte* pinned_pool_ = nullptr;
        std::size_t pinned_pool_bytes_ = 0;
        std::vector<std::byte*> free_buffers_;

        // LRU bookkeeping. lru_list_ has the LRU at front, MRU at back.
        mutable std::mutex mutex_;
        LruList lru_list_;
        std::unordered_map<std::size_t, LruList::iterator> map_;

        // Async flush worker: producer/consumer queue.
        // is_moments=true distinguishes a write_moments() job from a write_block().
        struct FlushJob {
            std::size_t block_id;
            std::vector<std::byte> payload; // Owned copy so the cache buffer can be reused immediately
            bool is_moments = false;
        };
        std::mutex flush_mutex_;
        std::condition_variable flush_cv_;          ///< Worker wakes on job arrival or stop
        std::condition_variable flush_drain_cv_;    ///< Notified when a block's in-flight count hits 0
        std::condition_variable flush_room_cv_;     ///< Notified when queue size drops below high-watermark
        std::list<FlushJob> flush_queue_;
        std::atomic<bool> flush_stop_{false};
        std::atomic<bool> worker_busy_{false};       ///< True while worker is processing a job
        // Per-block reference count of pending+in-flight async flush jobs. A block
        // with a non-zero entry has dirty bytes that have been evicted from cache
        // but may not yet be on the SSD; @ref get() must wait for the count to
        // hit 0 before issuing the BlockStore read on that block.
        std::unordered_map<std::size_t, std::uint32_t> in_flight_flush_counts_;
        std::string flush_worker_error_;             ///< Last worker error (under flush_mutex_)
        std::thread flush_thread_;

        std::unique_ptr<Stats> stats_;

        // Helpers (called with mutex_ held unless noted).
        std::byte* allocate_buffer_();
        void release_buffer_(std::byte* buf);
        std::expected<LruList::iterator, std::string> insert_block_(std::size_t block_id);
        void touch_(LruList::iterator it);
        void evict_one_clean_();
        // Phase 3.5.5: try to evict the LRU-most unpinned block, async-flushing
        // any dirty regions. Returns true if a block was evicted. Must be called
        // with mutex_ held; briefly acquires flush_mutex_ in addition
        // (lock order: mutex_ -> flush_mutex_) to atomically bump in_flight_
        // counts before the cache entry is removed.
        bool evict_one_async_();
        // Block while an async flush is in flight for @p block_id. Caller MUST NOT
        // hold mutex_. Increments stats_->miss_drain_waits if it actually waited.
        void wait_for_in_flight_(std::size_t block_id);
        void flush_worker_();
    };

} // namespace lfs::core
