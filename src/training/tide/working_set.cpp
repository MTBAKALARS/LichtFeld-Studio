/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file working_set.cpp
 * @brief Phase 2.5 GPU working set: A/B double-buffer + async prefetch.
 *
 * Two device buffers ("A" at index 0, "B" at index 1). At any time, one is
 * the "active" buffer (read by the trainer) and the other is the "inactive"
 * buffer (written by the prefetch stream).
 *
 *   prefetch(block_ids):
 *     1. Compute retained = block_ids \cap active_resident.
 *     2. Compute incoming = block_ids \ active_resident.
 *     3. Reset the inactive buffer's bookkeeping (all slots free).
 *     4. Assign each retained id a slot in the inactive buffer and issue a
 *        D2D cudaMemcpyAsync from active[old_slot] to inactive[new_slot].
 *     5. For each incoming id: cache.get() -> H2D cudaMemcpyAsync into a free
 *        slot of the inactive buffer. Track pinned ids for later unpin.
 *     6. cudaEventRecord(prefetch_done_, stream).
 *
 *   wait_and_activate():
 *     1. cudaEventSynchronize(prefetch_done_).
 *     2. cache.unpin() every block pinned during prefetch.
 *     3. Build the new slice table in caller-requested order.
 *     4. Swap active_idx_.
 *     5. Update stats.
 *
 * Concurrency: trainer kernels read the active buffer on the default stream
 * while prefetch writes to the inactive buffer on the prefetch stream. The
 * two buffers are disjoint device allocations, so no cross-stream sync is
 * required between training and prefetch. The swap in wait_and_activate()
 * happens after the host has explicitly synchronized on the prefetch event.
 */

#include "tide/working_set.hpp"

#include "core/logger.hpp"
#include "core/tiered_cache.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace lfs::training::tide {

    namespace {

        std::string cuda_err(std::string_view what, cudaError_t e) {
            return std::string(what) + ": " + cudaGetErrorString(e);
        }

    } // namespace

    struct WorkingSet::Impl {
        // Two device buffers, each `capacity_blocks * slot_bytes` bytes,
        // where slot_bytes = bytes_per_block + moments_bytes_per_block.
        // active_idx selects which one the trainer reads via device_buffer().
        std::byte* device_buffer[2] = {nullptr, nullptr};
        int active_idx = 0;

        cudaStream_t stream = nullptr;            ///< Dedicated H2D/D2D stream
        cudaEvent_t  prefetch_done = nullptr;     ///< Recorded at end of prefetch()
        bool         prefetch_pending = false;    ///< True between prefetch() and wait_and_activate()

        // Per-buffer slot ownership. Each entry is a resident map (block_id -> local slot)
        // and a free-slot LIFO. Indexed by buffer index (0 or 1).
        std::unordered_map<std::size_t, std::size_t> resident_[2];
        std::vector<std::size_t> free_slots_[2];

        // Per-buffer per-slot dirty bits. Sized capacity_blocks each. Carried
        // forward to dst on retain; cleared in dst on H2D upload (cache copy is
        // canonical for fresh blocks); cleared in src after writeback in prefetch.
        std::vector<std::uint8_t> data_dirty_[2];
        std::vector<std::uint8_t> moments_dirty_[2];

        // Slice table for the currently-active buffer (rebuilt at activate time).
        std::vector<BlockSlice> active_slices_;

        // The block_ids passed to the most recent prefetch(); used to build the
        // slice table in caller-requested order at activate time.
        std::vector<std::size_t> pending_block_ids_;

        // Block ids pinned via TieredCache::get() during prefetch(). Released
        // (unpinned, not dirty) inside wait_and_activate() after the H2D event fires.
        std::vector<std::size_t> pending_unpins_;

        // Block ids pinned via TieredCache::pin_for_write()/pin_moments_for_write()
        // during a prefetch-time writeback. Released (unpinned, dirty=false because
        // the pin call itself already marked the cache slot dirty) after the D2H
        // event fires in wait_and_activate().
        std::vector<std::size_t> pending_writeback_unpins_;

        // Non-owning pointer to the cache that pinned `pending_unpins_`. Set by
        // prefetch(), consumed by wait_and_activate(). Null when no prefetch is in flight.
        lfs::core::TieredCache* pending_cache = nullptr;
    };

    WorkingSet::~WorkingSet() {
        if (!impl_) return;
        if (impl_->prefetch_done != nullptr) {
            cudaEventDestroy(impl_->prefetch_done);
            impl_->prefetch_done = nullptr;
        }
        if (impl_->stream != nullptr) {
            cudaStreamDestroy(impl_->stream);
            impl_->stream = nullptr;
        }
        for (int i = 0; i < 2; ++i) {
            if (impl_->device_buffer[i] != nullptr) {
                cudaFree(impl_->device_buffer[i]);
                impl_->device_buffer[i] = nullptr;
            }
        }
    }

    WorkingSet::WorkingSet(WorkingSet&&) noexcept = default;
    WorkingSet& WorkingSet::operator=(WorkingSet&&) noexcept = default;

    // ============================================================
    // create
    // ============================================================
    std::expected<std::unique_ptr<WorkingSet>, std::string>
    WorkingSet::create(const Config& config) {
        if (config.capacity_blocks == 0) {
            return std::unexpected<std::string>("WorkingSet::create: capacity_blocks must be > 0");
        }
        if (config.bytes_per_block == 0) {
            return std::unexpected<std::string>("WorkingSet::create: bytes_per_block must be > 0");
        }

        if (auto e = cudaSetDevice(config.cuda_device); e != cudaSuccess) {
            return std::unexpected<std::string>(cuda_err("WorkingSet::create cudaSetDevice", e));
        }

        auto ws = std::unique_ptr<WorkingSet>(new WorkingSet());
        ws->config_ = config;
        ws->impl_ = std::make_unique<Impl>();

        const std::size_t slot_bytes = config.bytes_per_block + config.moments_bytes_per_block;
        const std::size_t total_bytes = config.capacity_blocks * slot_bytes;

        for (int i = 0; i < 2; ++i) {
            if (auto e = cudaMalloc(reinterpret_cast<void**>(&ws->impl_->device_buffer[i]),
                                    total_bytes);
                e != cudaSuccess) {
                return std::unexpected<std::string>(cuda_err(
                    "WorkingSet::create cudaMalloc(buffer " + std::to_string(i) + ", " +
                        std::to_string(total_bytes) + " B)",
                    e));
            }
        }

        if (auto e = cudaStreamCreateWithFlags(&ws->impl_->stream, cudaStreamNonBlocking);
            e != cudaSuccess) {
            return std::unexpected<std::string>(cuda_err("WorkingSet::create cudaStreamCreate", e));
        }
        if (auto e = cudaEventCreateWithFlags(&ws->impl_->prefetch_done,
                                              cudaEventDisableTiming);
            e != cudaSuccess) {
            return std::unexpected<std::string>(cuda_err("WorkingSet::create cudaEventCreate", e));
        }

        // Initialize free-slot stacks and zero-init per-slot dirty bits for both buffers.
        for (int i = 0; i < 2; ++i) {
            ws->impl_->free_slots_[i].reserve(config.capacity_blocks);
            for (std::size_t s = config.capacity_blocks; s-- > 0;) {
                ws->impl_->free_slots_[i].push_back(s);
            }
            ws->impl_->data_dirty_[i].assign(config.capacity_blocks, 0);
            ws->impl_->moments_dirty_[i].assign(config.capacity_blocks, 0);
        }

        if (config.moments_bytes_per_block > 0) {
            LOG_INFO(
                "WorkingSet::create A/B buffers, capacity_blocks={} slot={} B "
                "(data {} + moments {}), total_vram={:.2f} MiB",
                config.capacity_blocks, slot_bytes,
                config.bytes_per_block, config.moments_bytes_per_block,
                2.0 * static_cast<double>(total_bytes) / (1ULL << 20));
        } else {
            LOG_INFO(
                "WorkingSet::create A/B buffers, capacity_blocks={} bytes_per_block={} "
                "total_vram={:.2f} MiB",
                config.capacity_blocks, config.bytes_per_block,
                // Two buffers, hence x2.
                2.0 * static_cast<double>(total_bytes) / (1ULL << 20));
        }

        return ws;
    }

    // ============================================================
    // prefetch
    // ============================================================
    std::expected<void, std::string>
    WorkingSet::prefetch(lfs::core::TieredCache& cache,
                         std::span<const std::size_t> block_ids) {
        if (!impl_) {
            return std::unexpected<std::string>("WorkingSet::prefetch: moved-from instance");
        }
        if (impl_->prefetch_pending) {
            return std::unexpected<std::string>(
                "WorkingSet::prefetch: previous prefetch not yet activated "
                "(call wait_and_activate() first)");
        }
        if (block_ids.size() > config_.capacity_blocks) {
            return std::unexpected<std::string>(
                "WorkingSet::prefetch: requested " + std::to_string(block_ids.size()) +
                " blocks but capacity is " + std::to_string(config_.capacity_blocks));
        }
        // Cross-check moments size with what the cache reports. We only allow
        // configurations where the WS slot stride matches the cache slot stride.
        if (config_.moments_bytes_per_block != cache.moments_bytes_per_block()) {
            return std::unexpected<std::string>(
                "WorkingSet::prefetch: moments_bytes_per_block mismatch (WS=" +
                std::to_string(config_.moments_bytes_per_block) +
                ", cache=" + std::to_string(cache.moments_bytes_per_block()) + ")");
        }

        if (auto e = cudaSetDevice(config_.cuda_device); e != cudaSuccess) {
            return std::unexpected<std::string>(cuda_err("prefetch cudaSetDevice", e));
        }

        const int  src_idx  = impl_->active_idx;
        const int  dst_idx  = 1 - src_idx;
        const auto bpb      = config_.bytes_per_block;
        const auto mpb      = config_.moments_bytes_per_block;
        const auto slot     = bpb + mpb;
        const bool has_mom  = (mpb > 0);

        // === Build a set of new ids for fast "is this id retained?" lookup ===
        std::unordered_map<std::size_t, std::size_t> new_set_index;
        new_set_index.reserve(block_ids.size());
        for (std::size_t i = 0; i < block_ids.size(); ++i) {
            new_set_index.emplace(block_ids[i], i);
        }

        impl_->pending_unpins_.clear();
        impl_->pending_writeback_unpins_.clear();
        impl_->pending_cache = &cache;

        std::uint64_t writeback_count = 0;
        std::uint64_t writeback_bytes = 0;

        // === Writeback pass: evict any dirty src-buffer slot not in the new set ===
        // Each writeback pins the cache slot for write, then D2H-copies the dirty
        // region(s) directly into the cache's pinned host buffer. The unpins fire
        // in wait_and_activate after the prefetch event syncs.
        for (auto& [src_id, src_slot] : impl_->resident_[src_idx]) {
            if (new_set_index.find(src_id) != new_set_index.end()) continue; // retained, no evict
            const bool d_dirty = impl_->data_dirty_[src_idx][src_slot] != 0;
            const bool m_dirty = has_mom && impl_->moments_dirty_[src_idx][src_slot] != 0;
            if (!d_dirty && !m_dirty) continue;

            std::byte* src = impl_->device_buffer[src_idx] + src_slot * slot;

            if (d_dirty) {
                auto pin = cache.pin_for_write(src_id);
                if (!pin.has_value()) {
                    return std::unexpected<std::string>(
                        "WorkingSet::prefetch: cache.pin_for_write(" + std::to_string(src_id) +
                        ") failed: " + pin.error());
                }
                const auto e = cudaMemcpyAsync(pin.value().data(), src, bpb,
                                               cudaMemcpyDeviceToHost, impl_->stream);
                if (e != cudaSuccess) {
                    cache.unpin(src_id, /*dirty=*/false);
                    return std::unexpected<std::string>(
                        cuda_err("cudaMemcpyAsync D2H data block " + std::to_string(src_id), e));
                }
                writeback_bytes += bpb;
            }
            if (m_dirty) {
                if (!d_dirty) {
                    // Need a pin first if we didn't take one above.
                    auto pin = cache.get(src_id);
                    if (!pin.has_value()) {
                        return std::unexpected<std::string>(
                            "WorkingSet::prefetch: cache.get(" + std::to_string(src_id) +
                            ") for moments writeback failed: " + pin.error());
                    }
                }
                auto mpin = cache.pin_moments_for_write(src_id);
                if (!mpin.has_value()) {
                    cache.unpin(src_id, /*dirty=*/false);
                    return std::unexpected<std::string>(
                        "WorkingSet::prefetch: cache.pin_moments_for_write(" +
                        std::to_string(src_id) + ") failed: " + mpin.error());
                }
                const auto e = cudaMemcpyAsync(mpin.value().data(), src + bpb, mpb,
                                               cudaMemcpyDeviceToHost, impl_->stream);
                if (e != cudaSuccess) {
                    cache.unpin(src_id, /*dirty=*/false);
                    return std::unexpected<std::string>(
                        cuda_err("cudaMemcpyAsync D2H moments block " + std::to_string(src_id), e));
                }
                writeback_bytes += mpb;
            }
            impl_->pending_writeback_unpins_.push_back(src_id);
            ++writeback_count;
        }

        // === Reset destination bookkeeping ===
        impl_->resident_[dst_idx].clear();
        impl_->free_slots_[dst_idx].clear();
        impl_->free_slots_[dst_idx].reserve(config_.capacity_blocks);
        for (std::size_t s = config_.capacity_blocks; s-- > 0;) {
            impl_->free_slots_[dst_idx].push_back(s);
        }
        std::fill(impl_->data_dirty_[dst_idx].begin(), impl_->data_dirty_[dst_idx].end(), 0);
        std::fill(impl_->moments_dirty_[dst_idx].begin(), impl_->moments_dirty_[dst_idx].end(), 0);

        // Remember the request so wait_and_activate() can rebuild the slice
        // table in the order the caller specified.
        impl_->pending_block_ids_.assign(block_ids.begin(), block_ids.end());

        std::uint64_t retained_count = 0;
        std::uint64_t retained_bytes = 0;
        std::uint64_t uploaded_count = 0;
        std::uint64_t uploaded_bytes = 0;

        // === Issue D2D + H2D copies into dst buffer ===
        for (std::size_t id : block_ids) {
            // Acquire a destination slot.
            if (impl_->free_slots_[dst_idx].empty()) {
                return std::unexpected<std::string>(
                    "WorkingSet::prefetch: ran out of dst slots (logic bug, "
                    "block_ids.size() should be <= capacity)");
            }
            const std::size_t dst_slot = impl_->free_slots_[dst_idx].back();
            impl_->free_slots_[dst_idx].pop_back();
            std::byte* dst = impl_->device_buffer[dst_idx] + dst_slot * slot;

            // Retain path: block is already resident in src buffer.
            const auto src_it = impl_->resident_[src_idx].find(id);
            if (src_it != impl_->resident_[src_idx].end()) {
                std::byte* src = impl_->device_buffer[src_idx] + src_it->second * slot;
                // Single D2D copy of the full slot (data + moments).
                const auto e = cudaMemcpyAsync(dst, src, slot,
                                               cudaMemcpyDeviceToDevice, impl_->stream);
                if (e != cudaSuccess) {
                    impl_->free_slots_[dst_idx].push_back(dst_slot);
                    return std::unexpected<std::string>(
                        cuda_err("cudaMemcpyAsync D2D block " + std::to_string(id), e));
                }
                // Carry forward dirty bits (a retained dirty block stays dirty
                // in the new active buffer; the writeback already happened above
                // only for blocks that were NOT retained).
                impl_->data_dirty_[dst_idx][dst_slot]    = impl_->data_dirty_[src_idx][src_it->second];
                impl_->moments_dirty_[dst_idx][dst_slot] = impl_->moments_dirty_[src_idx][src_it->second];
                impl_->resident_[dst_idx].emplace(id, dst_slot);
                ++retained_count;
                retained_bytes += slot;
                continue;
            }

            // Upload path: block is not resident; pin host page and H2D-copy.
            auto host_view = cache.get(id);
            if (!host_view.has_value()) {
                impl_->free_slots_[dst_idx].push_back(dst_slot);
                return std::unexpected<std::string>(
                    "WorkingSet::prefetch: cache.get(" + std::to_string(id) +
                    ") failed: " + host_view.error());
            }
            const auto bytes = host_view.value().size();
            if (bytes != bpb) {
                cache.unpin(id, /*dirty=*/false);
                impl_->free_slots_[dst_idx].push_back(dst_slot);
                return std::unexpected<std::string>(
                    "WorkingSet::prefetch: block " + std::to_string(id) +
                    " size " + std::to_string(bytes) +
                    " != bytes_per_block " + std::to_string(bpb));
            }

            const auto e = cudaMemcpyAsync(dst, host_view.value().data(), bpb,
                                           cudaMemcpyHostToDevice, impl_->stream);
            if (e != cudaSuccess) {
                cache.unpin(id, /*dirty=*/false);
                impl_->free_slots_[dst_idx].push_back(dst_slot);
                return std::unexpected<std::string>(
                    cuda_err("cudaMemcpyAsync H2D block " + std::to_string(id), e));
            }
            uploaded_bytes += bpb;

            // Also pull the moments region (eagerly populated in cache by get()).
            if (has_mom) {
                auto mview = cache.get_moments(id);
                if (!mview.has_value()) {
                    cache.unpin(id, /*dirty=*/false);
                    impl_->free_slots_[dst_idx].push_back(dst_slot);
                    return std::unexpected<std::string>(
                        "WorkingSet::prefetch: cache.get_moments(" + std::to_string(id) +
                        ") failed: " + mview.error());
                }
                if (mview.value().size() != mpb) {
                    cache.unpin(id, /*dirty=*/false);
                    impl_->free_slots_[dst_idx].push_back(dst_slot);
                    return std::unexpected<std::string>(
                        "WorkingSet::prefetch: block " + std::to_string(id) +
                        " moments size " + std::to_string(mview.value().size()) +
                        " != moments_bytes_per_block " + std::to_string(mpb));
                }
                const auto e2 = cudaMemcpyAsync(dst + bpb, mview.value().data(), mpb,
                                                cudaMemcpyHostToDevice, impl_->stream);
                if (e2 != cudaSuccess) {
                    cache.unpin(id, /*dirty=*/false);
                    impl_->free_slots_[dst_idx].push_back(dst_slot);
                    return std::unexpected<std::string>(
                        cuda_err("cudaMemcpyAsync H2D moments block " + std::to_string(id), e2));
                }
                uploaded_bytes += mpb;
            }

            impl_->resident_[dst_idx].emplace(id, dst_slot);
            // Fresh upload — dst slot starts clean.
            impl_->data_dirty_[dst_idx][dst_slot]    = 0;
            impl_->moments_dirty_[dst_idx][dst_slot] = 0;
            impl_->pending_unpins_.push_back(id);  // unpin after wait_and_activate
            ++uploaded_count;
        }

        // Record completion event so wait_and_activate can sync on it.
        if (auto e = cudaEventRecord(impl_->prefetch_done, impl_->stream); e != cudaSuccess) {
            // Best-effort: try to release any pinned host pages we acquired before the failure.
            for (auto pinned_id : impl_->pending_unpins_) cache.unpin(pinned_id, /*dirty=*/false);
            for (auto pinned_id : impl_->pending_writeback_unpins_) cache.unpin(pinned_id, /*dirty=*/false);
            impl_->pending_unpins_.clear();
            impl_->pending_writeback_unpins_.clear();
            impl_->pending_block_ids_.clear();
            impl_->pending_cache = nullptr;
            return std::unexpected<std::string>(cuda_err("cudaEventRecord(prefetch_done)", e));
        }

        impl_->prefetch_pending = true;
        stats_.blocks_uploaded     += uploaded_count;
        stats_.bytes_uploaded      += uploaded_bytes;
        stats_.blocks_retained     += retained_count;
        stats_.bytes_d2d_copied    += retained_bytes;
        stats_.blocks_written_back += writeback_count;
        stats_.bytes_written_back  += writeback_bytes;
        ++stats_.prefetches;
        return {};
    }

    // ============================================================
    // wait_and_activate
    // ============================================================
    std::expected<void, std::string>
    WorkingSet::wait_and_activate() {
        if (!impl_) {
            return std::unexpected<std::string>("WorkingSet::wait_and_activate: moved-from instance");
        }
        if (!impl_->prefetch_pending) {
            return std::unexpected<std::string>(
                "WorkingSet::wait_and_activate: no prefetch in flight");
        }

        if (auto e = cudaEventSynchronize(impl_->prefetch_done); e != cudaSuccess) {
            return std::unexpected<std::string>(
                cuda_err("cudaEventSynchronize(prefetch_done)", e));
        }

        // Release pinned host pages now that the H2D / D2H events have fired.
        // Upload pins: data flowed cache->WS, cache slot stays clean.
        // Writeback pins: data flowed WS->cache, the pin_*_for_write call already
        //   marked the cache slot dirty so unpin's dirty arg can be false.
        if (impl_->pending_cache != nullptr) {
            for (auto id : impl_->pending_unpins_) {
                impl_->pending_cache->unpin(id, /*dirty=*/false);
            }
            for (auto id : impl_->pending_writeback_unpins_) {
                impl_->pending_cache->unpin(id, /*dirty=*/false);
            }
        }
        impl_->pending_unpins_.clear();
        impl_->pending_writeback_unpins_.clear();
        impl_->pending_cache = nullptr;

        // Build slice table in caller-requested order, from the freshly-staged
        // (dst) buffer that's about to become active.
        const int new_active = 1 - impl_->active_idx;
        impl_->active_slices_.clear();
        impl_->active_slices_.reserve(impl_->pending_block_ids_.size());
        for (auto id : impl_->pending_block_ids_) {
            auto it = impl_->resident_[new_active].find(id);
            if (it == impl_->resident_[new_active].end()) {
                return std::unexpected<std::string>(
                    "WorkingSet::wait_and_activate: block " + std::to_string(id) +
                    " missing from staged buffer (logic bug)");
            }
            impl_->active_slices_.push_back(BlockSlice{id, it->second});
        }
        impl_->pending_block_ids_.clear();

        // Swap.
        impl_->active_idx = new_active;
        impl_->prefetch_pending = false;
        ++stats_.activates;
        return {};
    }

    bool WorkingSet::prefetch_pending() const noexcept {
        return impl_ && impl_->prefetch_pending;
    }

    // ============================================================
    // load_and_activate (synchronous convenience)
    // ============================================================
    std::expected<void, std::string>
    WorkingSet::load_and_activate(lfs::core::TieredCache& cache,
                                  std::span<const std::size_t> block_ids) {
        if (auto r = prefetch(cache, block_ids); !r.has_value()) {
            return r;
        }
        if (auto r = wait_and_activate(); !r.has_value()) {
            return r;
        }
        ++stats_.loads;
        return {};
    }

    // ============================================================
    // Accessors
    // ============================================================
    const void* WorkingSet::device_buffer() const noexcept {
        return impl_ ? impl_->device_buffer[impl_->active_idx] : nullptr;
    }

    void* WorkingSet::mutable_device_buffer() noexcept {
        return impl_ ? impl_->device_buffer[impl_->active_idx] : nullptr;
    }

    float* WorkingSet::moments_device_buffer(std::size_t local_idx) noexcept {
        if (!impl_) return nullptr;
        if (config_.moments_bytes_per_block == 0) return nullptr;
        if (local_idx >= config_.capacity_blocks) return nullptr;
        const std::size_t slot = config_.bytes_per_block + config_.moments_bytes_per_block;
        std::byte* base = impl_->device_buffer[impl_->active_idx];
        return reinterpret_cast<float*>(base + local_idx * slot + config_.bytes_per_block);
    }

    const float* WorkingSet::moments_device_buffer(std::size_t local_idx) const noexcept {
        if (!impl_) return nullptr;
        if (config_.moments_bytes_per_block == 0) return nullptr;
        if (local_idx >= config_.capacity_blocks) return nullptr;
        const std::size_t slot = config_.bytes_per_block + config_.moments_bytes_per_block;
        const std::byte* base = impl_->device_buffer[impl_->active_idx];
        return reinterpret_cast<const float*>(base + local_idx * slot + config_.bytes_per_block);
    }

    void WorkingSet::mark_dirty(std::size_t local_idx, bool data_dirty, bool moments_dirty) {
        if (!impl_) return;
        if (local_idx >= config_.capacity_blocks) return;
        if (data_dirty)    impl_->data_dirty_[impl_->active_idx][local_idx]    = 1;
        if (moments_dirty && config_.moments_bytes_per_block > 0) {
            impl_->moments_dirty_[impl_->active_idx][local_idx] = 1;
        }
    }

    std::span<const WorkingSet::BlockSlice> WorkingSet::active_slices() const noexcept {
        if (!impl_) return {};
        return std::span<const BlockSlice>(impl_->active_slices_);
    }

    std::size_t WorkingSet::active_block_count() const noexcept {
        return impl_ ? impl_->active_slices_.size() : 0;
    }

    std::size_t WorkingSet::active_gaussian_count() const noexcept {
        if (!impl_) return 0;
        // Per-block gaussian count derives from the parameter region only.
        // Tail-padded blocks contribute their full block_size; callers that need
        // the precise live count must intersect with BlockStore::lookup.
        const std::size_t per_block_gauss =
            config_.bytes_per_block / lfs::core::BlockStore::kBytesPerGaussian;
        return impl_->active_slices_.size() * per_block_gauss;
    }

} // namespace lfs::training::tide
