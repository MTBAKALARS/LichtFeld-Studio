/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file working_set.cpp
 * @brief V1 GPU working set: synchronous H2D, packed CACHE-layout VRAM buffer.
 *
 * The trainer hands us a list of visible block_ids per iteration. We:
 *   1. Resolve which incoming blocks are already resident (retain them).
 *   2. Free slots whose current occupants are no longer visible.
 *   3. cudaMemcpyAsync the new arrivals from pinned host (via TieredCache::get)
 *      into the freed slots.
 *   4. cudaStreamSynchronize to make the buffer visible to the next kernel.
 *
 * No CUDA kernels are launched here — only the host-side cudaMemcpyAsync /
 * cudaStreamSynchronize APIs. This keeps the TU as a .cpp (no nvcc).
 *
 * The double-buffer / overlap-aware variant (paper Algorithm 1) lands in a
 * follow-up commit; the V1 isolates the data plumbing for end-to-end validation.
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
        std::byte* device_buffer = nullptr;   ///< capacity_blocks * bytes_per_block bytes on GPU
        cudaStream_t stream = nullptr;        ///< Dedicated H2D stream

        // Slot ownership.
        // Map block_id -> local_index. Free slots are not in the map; reuse via free_slots_.
        std::unordered_map<std::size_t, std::size_t> resident_;
        std::vector<std::size_t> free_slots_; ///< LIFO stack of free local indices

        // Slice table rebuilt on every load_and_activate.
        std::vector<BlockSlice> active_slices_;
    };

    WorkingSet::~WorkingSet() {
        if (!impl_) return;
        if (impl_->device_buffer != nullptr) {
            cudaFree(impl_->device_buffer);
            impl_->device_buffer = nullptr;
        }
        if (impl_->stream != nullptr) {
            cudaStreamDestroy(impl_->stream);
            impl_->stream = nullptr;
        }
    }

    WorkingSet::WorkingSet(WorkingSet&&) noexcept = default;
    WorkingSet& WorkingSet::operator=(WorkingSet&&) noexcept = default;

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

        const std::size_t total_bytes = config.capacity_blocks * config.bytes_per_block;
        if (auto e = cudaMalloc(reinterpret_cast<void**>(&ws->impl_->device_buffer), total_bytes);
            e != cudaSuccess) {
            return std::unexpected<std::string>(
                cuda_err("WorkingSet::create cudaMalloc(" + std::to_string(total_bytes) + " B)", e));
        }

        if (auto e = cudaStreamCreateWithFlags(&ws->impl_->stream, cudaStreamNonBlocking);
            e != cudaSuccess) {
            cudaFree(ws->impl_->device_buffer);
            ws->impl_->device_buffer = nullptr;
            return std::unexpected<std::string>(cuda_err("WorkingSet::create cudaStreamCreate", e));
        }

        // Initialize free-slot stack with every slot.
        ws->impl_->free_slots_.reserve(config.capacity_blocks);
        for (std::size_t i = config.capacity_blocks; i-- > 0;) {
            ws->impl_->free_slots_.push_back(i);
        }

        LOG_INFO("WorkingSet::create capacity_blocks={} bytes_per_block={} total_vram={:.2f} MiB",
                 config.capacity_blocks, config.bytes_per_block,
                 static_cast<double>(total_bytes) / (1ULL << 20));

        return ws;
    }

    std::expected<void, std::string>
    WorkingSet::load_and_activate(lfs::core::TieredCache& cache,
                                  std::span<const std::size_t> block_ids) {
        if (!impl_) {
            return std::unexpected<std::string>("WorkingSet::load_and_activate: moved-from instance");
        }
        if (block_ids.size() > config_.capacity_blocks) {
            return std::unexpected<std::string>(
                "WorkingSet::load_and_activate: requested " + std::to_string(block_ids.size()) +
                " blocks but capacity is " + std::to_string(config_.capacity_blocks));
        }

        if (auto e = cudaSetDevice(config_.cuda_device); e != cudaSuccess) {
            return std::unexpected<std::string>(cuda_err("cudaSetDevice", e));
        }

        // === Step 1: Determine retained vs incoming ===
        // Build a quick membership set of the requested block_ids.
        std::unordered_map<std::size_t, char> requested;
        requested.reserve(block_ids.size() * 2);
        for (auto id : block_ids) requested[id] = 1;

        // Step 1a: free slots whose blocks are no longer requested.
        std::vector<std::size_t> evicted;
        evicted.reserve(impl_->resident_.size());
        for (auto it = impl_->resident_.begin(); it != impl_->resident_.end();) {
            if (requested.find(it->first) == requested.end()) {
                impl_->free_slots_.push_back(it->second);
                evicted.push_back(it->first);
                it = impl_->resident_.erase(it);
            } else {
                ++it;
            }
        }

        // Step 1b: incoming = requested blocks not yet resident.
        std::vector<std::size_t> incoming;
        incoming.reserve(block_ids.size());
        std::size_t retained = 0;
        for (auto id : block_ids) {
            if (impl_->resident_.find(id) != impl_->resident_.end()) {
                ++retained;
            } else {
                incoming.push_back(id);
            }
        }

        // Sanity: incoming.size() must fit in free_slots_ (it does, because
        // requested.size() <= capacity and retained slots stay put).
        if (incoming.size() > impl_->free_slots_.size()) {
            return std::unexpected<std::string>(
                "WorkingSet::load_and_activate: internal slot accounting mismatch (need " +
                std::to_string(incoming.size()) + " free, have " +
                std::to_string(impl_->free_slots_.size()) + ")");
        }

        // === Step 2: H2D copy incoming blocks (synchronous on dedicated stream) ===
        for (std::size_t id : incoming) {
            auto host_view = cache.get(id);
            if (!host_view.has_value()) {
                return std::unexpected<std::string>(
                    "WorkingSet::load_and_activate: cache.get(" + std::to_string(id) +
                    ") failed: " + host_view.error());
            }
            const auto bytes = host_view.value().size();
            if (bytes != config_.bytes_per_block) {
                cache.unpin(id, /*dirty=*/false);
                return std::unexpected<std::string>(
                    "WorkingSet::load_and_activate: block " + std::to_string(id) +
                    " size " + std::to_string(bytes) + " != bytes_per_block " +
                    std::to_string(config_.bytes_per_block));
            }

            const std::size_t slot = impl_->free_slots_.back();
            impl_->free_slots_.pop_back();
            std::byte* dst = impl_->device_buffer + slot * config_.bytes_per_block;

            const auto e = cudaMemcpyAsync(dst, host_view.value().data(),
                                           config_.bytes_per_block,
                                           cudaMemcpyHostToDevice, impl_->stream);
            if (e != cudaSuccess) {
                cache.unpin(id, /*dirty=*/false);
                impl_->free_slots_.push_back(slot); // give it back
                return std::unexpected<std::string>(cuda_err(
                    "cudaMemcpyAsync block " + std::to_string(id), e));
            }

            impl_->resident_.emplace(id, slot);
            stats_.bytes_uploaded += config_.bytes_per_block;
            ++stats_.blocks_uploaded;

            // NOTE: We unpin only AFTER the stream sync below. The pinned host
            // buffer must remain valid until the H2D completes. We collect
            // pins to release in a second pass.
        }

        // Wait for all H2D to finish before unpinning.
        if (auto e = cudaStreamSynchronize(impl_->stream); e != cudaSuccess) {
            return std::unexpected<std::string>(cuda_err("cudaStreamSynchronize", e));
        }
        for (std::size_t id : incoming) {
            cache.unpin(id, /*dirty=*/false);
        }

        // === Step 3: Rebuild active_slices in the order the caller requested ===
        impl_->active_slices_.clear();
        impl_->active_slices_.reserve(block_ids.size());
        for (auto id : block_ids) {
            auto it = impl_->resident_.find(id);
            // We just made sure every requested id is resident; defensive check anyway.
            if (it == impl_->resident_.end()) {
                return std::unexpected<std::string>(
                    "WorkingSet::load_and_activate: block " + std::to_string(id) +
                    " not resident after load (logic bug)");
            }
            impl_->active_slices_.push_back(BlockSlice{id, it->second});
        }

        stats_.blocks_retained += retained;
        ++stats_.loads;
        return {};
    }

    const void* WorkingSet::device_buffer() const noexcept {
        return impl_ ? impl_->device_buffer : nullptr;
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
        // V1 assumes every resident block is a full block_size. The tail partial
        // block (if any) is padded with zeros at stream_ply_to_base time. Callers
        // that need the precise live count must intersect with BlockStore::lookup.
        const std::size_t per_block_gauss =
            config_.bytes_per_block / lfs::core::BlockStore::kBytesPerGaussian;
        return impl_->active_slices_.size() * per_block_gauss;
    }

} // namespace lfs::training::tide
