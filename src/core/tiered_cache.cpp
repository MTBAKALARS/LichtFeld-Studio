/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tiered_cache.hpp"

#include "core/logger.hpp"

#include <cstring>
#include <stdexcept>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace lfs::core {

    namespace {

        std::byte* alloc_pinned(std::size_t bytes) {
#if defined(_WIN32)
            // VirtualAlloc + VirtualLock pins pages in working set. CUDA cudaHostRegister
            // would be ideal once we link against CUDA in this TU; for now this gives
            // page-locked semantics suitable for OS DMA paths.
            void* p = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            if (!p) return nullptr;
            if (!VirtualLock(p, bytes)) {
                LOG_WARN("TieredCache: VirtualLock failed (size={} B); falling back to unlocked pages.", bytes);
                // Continue anyway — the pages are still valid.
            }
            return static_cast<std::byte*>(p);
#else
            void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED) return nullptr;
            if (::mlock(p, bytes) != 0) {
                LOG_WARN("TieredCache: mlock failed (size={} B); falling back to unlocked pages.", bytes);
            }
            return static_cast<std::byte*>(p);
#endif
        }

        void free_pinned(std::byte* p, std::size_t bytes) {
            if (!p) return;
#if defined(_WIN32)
            VirtualUnlock(p, bytes);
            VirtualFree(p, 0, MEM_RELEASE);
#else
            ::munlock(p, bytes);
            ::munmap(p, bytes);
#endif
        }

    } // namespace

    TieredCache::TieredCache(std::shared_ptr<BlockStore> store, const Config& config)
        : store_(std::move(store)),
          config_(config),
          bytes_per_block_(store_->bytes_per_block()),
          stats_(std::make_unique<Stats>()) {

        if (!store_) throw std::invalid_argument{"TieredCache: store must be non-null"};
        if (config_.capacity_blocks == 0) throw std::invalid_argument{"TieredCache: capacity_blocks must be > 0"};

        if (config_.pinned_pool_size_blocks == 0) {
            config_.pinned_pool_size_blocks = config_.capacity_blocks;
        }
        if (config_.pinned_pool_size_blocks < config_.capacity_blocks) {
            // We require enough buffers to back every resident block.
            config_.pinned_pool_size_blocks = config_.capacity_blocks;
        }

        pinned_pool_bytes_ = config_.pinned_pool_size_blocks * bytes_per_block_;
        pinned_pool_ = alloc_pinned(pinned_pool_bytes_);
        if (!pinned_pool_) {
            throw std::runtime_error{
                std::format("TieredCache: failed to allocate pinned pool ({} B = {:.2f} GiB)",
                            pinned_pool_bytes_,
                            static_cast<double>(pinned_pool_bytes_) / (1ull << 30))};
        }

        free_buffers_.reserve(config_.pinned_pool_size_blocks);
        for (std::size_t i = 0; i < config_.pinned_pool_size_blocks; ++i) {
            free_buffers_.push_back(pinned_pool_ + i * bytes_per_block_);
        }

        flush_thread_ = std::thread{[this] { flush_worker_(); }};

        LOG_INFO("TieredCache: capacity={} blocks, pinned pool={:.2f} GiB ({} buffers)",
                 config_.capacity_blocks,
                 static_cast<double>(pinned_pool_bytes_) / (1ull << 30),
                 config_.pinned_pool_size_blocks);
    }

    TieredCache::~TieredCache() {
        // Sync-flush dirty blocks before tearing down.
        if (auto r = flush_dirty(); !r) {
            LOG_ERROR("TieredCache::~TieredCache: flush_dirty failed: {}", r.error());
        }

        // Stop the worker.
        {
            std::scoped_lock lk(flush_mutex_);
            flush_stop_.store(true, std::memory_order_release);
        }
        flush_cv_.notify_all();
        if (flush_thread_.joinable()) flush_thread_.join();

        free_pinned(pinned_pool_, pinned_pool_bytes_);
    }

    // ============================================================
    // Read path
    // ============================================================

    std::expected<std::span<const std::byte>, std::string> TieredCache::get(std::size_t block_id) {
        std::unique_lock lk(mutex_);

        if (auto it = map_.find(block_id); it != map_.end()) {
            touch_(it->second);
            auto& node = *it->second->node;
            ++node.pin_count;
            stats_->hits.fetch_add(1, std::memory_order_relaxed);
            return std::span<const std::byte>{node.buffer, bytes_per_block_};
        }

        // Miss: insert (may evict), then synchronously read from store.
        auto inserted = insert_block_(block_id);
        if (!inserted) return std::unexpected{inserted.error()};

        auto& node = *(*inserted)->node;
        ++node.pin_count;
        stats_->misses.fetch_add(1, std::memory_order_relaxed);

        lk.unlock();
        // Disk read outside the lock — no other thread can touch a freshly-pinned node.
        if (auto r = store_->read_block(block_id, std::span<std::byte>{node.buffer, bytes_per_block_}); !r) {
            // Roll back the insertion on read failure.
            std::scoped_lock lk2(mutex_);
            auto map_it = map_.find(block_id);
            if (map_it != map_.end()) {
                release_buffer_(map_it->second->node->buffer);
                lru_list_.erase(map_it->second);
                map_.erase(map_it);
            }
            return std::unexpected{r.error()};
        }

        return std::span<const std::byte>{node.buffer, bytes_per_block_};
    }

    std::expected<std::span<std::byte>, std::string> TieredCache::pin_for_write(std::size_t block_id) {
        // Identical to get() but mark dirty up front so unpin doesn't need a flag.
        auto r = get(block_id);
        if (!r) return std::unexpected{r.error()};
        std::scoped_lock lk(mutex_);
        auto it = map_.find(block_id);
        if (it == map_.end()) {
            // Theoretically impossible — get() just inserted it and we hold a pin.
            return std::unexpected{"pin_for_write: block disappeared after get()"};
        }
        auto& node = *it->second->node;
        if (!node.dirty) {
            node.dirty = true;
            stats_->dirty_blocks_resident.fetch_add(1, std::memory_order_relaxed);
        }
        // Cast away const — the buffer was always writable; get() returned a const view by convention.
        return std::span<std::byte>{node.buffer, bytes_per_block_};
    }

    void TieredCache::unpin(std::size_t block_id, bool dirty) {
        std::scoped_lock lk(mutex_);
        auto it = map_.find(block_id);
        if (it == map_.end()) return; // Defensive — shouldn't happen.
        auto& node = *it->second->node;
        if (node.pin_count == 0) return;
        --node.pin_count;
        if (dirty && !node.dirty) {
            node.dirty = true;
            stats_->dirty_blocks_resident.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // ============================================================
    // Eviction
    // ============================================================

    std::size_t TieredCache::evict_clean(std::size_t k) {
        std::scoped_lock lk(mutex_);
        std::size_t evicted = 0;
        // Walk from LRU end, skipping pinned and dirty entries.
        auto it = lru_list_.begin();
        while (evicted < k && it != lru_list_.end()) {
            auto& node = *it->node;
            if (node.pin_count == 0 && !node.dirty) {
                release_buffer_(node.buffer);
                map_.erase(node.block_id);
                it = lru_list_.erase(it);
                ++evicted;
                stats_->evictions.fetch_add(1, std::memory_order_relaxed);
            } else {
                ++it;
            }
        }
        return evicted;
    }

    std::expected<void, std::string> TieredCache::flush_dirty() {
        std::vector<std::size_t> to_flush;
        std::vector<std::vector<std::byte>> payloads;
        {
            std::scoped_lock lk(mutex_);
            for (auto& entry : lru_list_) {
                auto& node = *entry.node;
                if (node.dirty && node.pin_count == 0) {
                    to_flush.push_back(node.block_id);
                    payloads.emplace_back(bytes_per_block_);
                    std::memcpy(payloads.back().data(), node.buffer, bytes_per_block_);
                    node.dirty = false;
                    stats_->dirty_blocks_resident.fetch_sub(1, std::memory_order_relaxed);
                }
            }
        }
        for (std::size_t i = 0; i < to_flush.size(); ++i) {
            if (auto r = store_->write_block(to_flush[i], payloads[i]); !r) {
                return std::unexpected{r.error()};
            }
            stats_->sync_flushes.fetch_add(1, std::memory_order_relaxed);
        }
        return {};
    }

    // ============================================================
    // Introspection
    // ============================================================

    std::size_t TieredCache::resident_blocks() const noexcept {
        std::scoped_lock lk(mutex_);
        return lru_list_.size();
    }

    // ============================================================
    // Private helpers
    // ============================================================

    std::byte* TieredCache::allocate_buffer_() {
        if (!free_buffers_.empty()) {
            auto* p = free_buffers_.back();
            free_buffers_.pop_back();
            return p;
        }
        return nullptr;
    }

    void TieredCache::release_buffer_(std::byte* buf) {
        if (buf) free_buffers_.push_back(buf);
    }

    std::expected<TieredCache::LruList::iterator, std::string>
    TieredCache::insert_block_(std::size_t block_id) {
        // Make space if needed.
        while (lru_list_.size() >= config_.capacity_blocks) {
            const std::size_t before = lru_list_.size();
            evict_one_clean_();
            if (lru_list_.size() == before) {
                // Couldn't evict anything — every resident block is pinned or dirty.
                // Force a synchronous flush of the LRU dirty block to make progress.
                bool flushed_any = false;
                for (auto it = lru_list_.begin(); it != lru_list_.end(); ++it) {
                    auto& node = *it->node;
                    if (node.pin_count == 0 && node.dirty) {
                        // Flush synchronously while we hold the lock — simpler than queueing here.
                        std::vector<std::byte> payload(bytes_per_block_);
                        std::memcpy(payload.data(), node.buffer, bytes_per_block_);
                        // Drop the lock during the SSD write.
                        mutex_.unlock();
                        auto r = store_->write_block(node.block_id, payload);
                        mutex_.lock();
                        if (!r) return std::unexpected{r.error()};
                        node.dirty = false;
                        stats_->dirty_blocks_resident.fetch_sub(1, std::memory_order_relaxed);
                        stats_->sync_flushes.fetch_add(1, std::memory_order_relaxed);
                        flushed_any = true;
                        break;
                    }
                }
                if (!flushed_any) {
                    return std::unexpected{"TieredCache: cache full and all blocks pinned (deadlock)"};
                }
                evict_one_clean_();
            }
        }

        auto* buf = allocate_buffer_();
        if (!buf) {
            return std::unexpected{"TieredCache: pinned pool exhausted (configuration error)"};
        }

        auto node = std::make_shared<Node>();
        node->block_id = block_id;
        node->buffer = buf;
        node->pin_count = 0;
        node->dirty = false;

        lru_list_.push_back(ListEntry{std::move(node)});
        auto list_it = std::prev(lru_list_.end());
        map_[block_id] = list_it;
        return list_it;
    }

    void TieredCache::touch_(LruList::iterator it) {
        // Splice to MRU end (constant time).
        lru_list_.splice(lru_list_.end(), lru_list_, it);
    }

    void TieredCache::evict_one_clean_() {
        for (auto it = lru_list_.begin(); it != lru_list_.end(); ++it) {
            auto& node = *it->node;
            if (node.pin_count == 0 && !node.dirty) {
                release_buffer_(node.buffer);
                map_.erase(node.block_id);
                lru_list_.erase(it);
                stats_->evictions.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
    }

    void TieredCache::enqueue_flush_(std::size_t block_id, std::byte* buffer) {
        FlushJob job;
        job.block_id = block_id;
        job.payload.assign(buffer, buffer + bytes_per_block_);
        {
            std::scoped_lock lk(flush_mutex_);
            flush_queue_.push_back(std::move(job));
        }
        flush_cv_.notify_one();
    }

    void TieredCache::flush_worker_() {
        for (;;) {
            FlushJob job;
            {
                std::unique_lock lk(flush_mutex_);
                flush_cv_.wait(lk, [&] {
                    return flush_stop_.load(std::memory_order_acquire) || !flush_queue_.empty();
                });
                if (flush_stop_.load(std::memory_order_acquire) && flush_queue_.empty()) return;
                job = std::move(flush_queue_.front());
                flush_queue_.pop_front();
            }
            if (auto r = store_->write_block(job.block_id, job.payload); !r) {
                LOG_ERROR("TieredCache: async flush of block {} failed: {}", job.block_id, r.error());
            } else {
                stats_->async_flush_jobs.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

} // namespace lfs::core
