/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/block_store.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <vector>

// Forward declarations to keep CUDA out of the public header.
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t; // NOLINT(modernize-use-using)

namespace lfs::core {
    class TieredCache;
}

namespace lfs::training::tide {

    /**
     * @brief GPU-resident packed view of a working set of Gaussian blocks.
     *
     * Implements the VRAM tier of TideGS's SSD-CPU-GPU hierarchy (arXiv:2605.20150
     * Sec. 3.5). A WorkingSet owns **two** GPU buffers (A/B double-buffer), each
     * sized for @ref capacity_blocks blocks of `BlockStore::bytes_per_block()` bytes,
     * plus a per-buffer `block_to_local_slice` mapping that lets the rasterizer find
     * a specific block's contiguous slice in the active buffer.
     *
     * **Phase 2.5 (this implementation): A/B double-buffer + async prefetch.**
     * `prefetch()` stages the next resident set into the inactive buffer on a
     * dedicated CUDA stream and returns immediately. `wait_and_activate()` blocks
     * on the prefetch completion event and atomically swaps which buffer is active.
     * Retained blocks (still visible across frames) are copied D2D from the old
     * active buffer to the new one — no host round-trip. Only fresh arrivals incur
     * an H2D `cudaMemcpyAsync` from pinned host memory.
     *
     * The synchronous helper `load_and_activate()` is preserved as a convenience
     * (= `prefetch()` + `wait_and_activate()`) for code paths that don't overlap
     * training with prefetch.
     *
     * Memory layout in each buffer (CACHE order, identical to the BlockStore
     * on-disk format):
     *
     *     [block_0 (4096 Gaussians × 236 B)] [block_1 ...] ... [block_{N-1} ...]
     *
     * Per-Gaussian byte layout is fixed by @ref BlockStore::kBytesPerGaussian = 236.
     *
     * Thread-safety: not safe for concurrent calls. The trainer owns one WorkingSet
     * and drives it from the main training thread. Concurrency between prefetch
     * (writes to inactive buffer on prefetch stream) and trainer kernels (reads
     * from active buffer on the default stream) is safe because they touch disjoint
     * device memory.
     */
    class WorkingSet {
    public:
        /// One contiguous slice of the active buffer that holds a particular block's bytes.
        struct BlockSlice {
            std::size_t block_id = 0;
            std::size_t local_index = 0; ///< 0..capacity_blocks-1; multiply by bytes_per_block for byte offset
        };

        struct Config {
            std::size_t capacity_blocks = 0;     ///< Hard cap on resident block count per buffer in VRAM
            std::size_t bytes_per_block = 0;     ///< Must match BlockStore::bytes_per_block()
            int cuda_device = 0;                 ///< GPU index for cudaSetDevice
        };

        struct Stats {
            std::uint64_t loads            = 0;  ///< Number of completed load_and_activate cycles
            std::uint64_t prefetches       = 0;  ///< Number of prefetch() calls issued
            std::uint64_t activates        = 0;  ///< Number of wait_and_activate() calls completed
            std::uint64_t blocks_uploaded  = 0;  ///< Blocks H2D-copied from pinned host
            std::uint64_t bytes_uploaded   = 0;  ///< Bytes H2D-copied from pinned host
            std::uint64_t blocks_retained  = 0;  ///< Blocks already resident from prior frame (D2D-copied)
            std::uint64_t bytes_d2d_copied = 0;  ///< Bytes D2D-copied for retention across the A/B swap
        };

        ~WorkingSet();
        WorkingSet(const WorkingSet&) = delete;
        WorkingSet& operator=(const WorkingSet&) = delete;
        WorkingSet(WorkingSet&&) noexcept;
        WorkingSet& operator=(WorkingSet&&) noexcept;

        /**
         * @brief Allocate the two GPU buffers, the prefetch stream, and a completion event.
         *
         * @return WorkingSet on success; descriptive error on cudaMalloc/stream failure.
         */
        static std::expected<std::unique_ptr<WorkingSet>, std::string>
        create(const Config& config);

        /**
         * @brief Stage the next resident set into the **inactive** GPU buffer (async).
         *
         * For each id in @p block_ids:
         *   - If already resident in the active buffer → D2D-copy bytes into the
         *     inactive buffer (no host round-trip).
         *   - Otherwise → pin via `cache.get()` and H2D `cudaMemcpyAsync` into the
         *     inactive buffer.
         *
         * All copies run on the dedicated prefetch stream; the function records a
         * cudaEvent and returns without blocking. Call @ref wait_and_activate to
         * publish the new buffer.
         *
         * Note: it is an error to call prefetch() twice without an intervening
         * wait_and_activate(). Callers that want to abandon a prefetch must
         * complete it via wait_and_activate() first.
         *
         * @param cache Source of pinned-host block bytes. Pinned ids stay pinned
         *              until wait_and_activate() releases them.
         * @param block_ids Unique list of block_ids that must be resident after
         *                  the next wait_and_activate(). `size() <= capacity_blocks`.
         */
        std::expected<void, std::string>
        prefetch(lfs::core::TieredCache& cache,
                 std::span<const std::size_t> block_ids);

        /**
         * @brief Block on the pending prefetch, then atomically swap A/B.
         *
         * After return, @ref device_buffer / @ref active_slices reflect the set
         * staged by the most recent prefetch(). Pinned host pages from that
         * prefetch are released.
         *
         * Returns an error if no prefetch is currently in flight, or if the
         * `cudaEventSynchronize` fails.
         */
        std::expected<void, std::string>
        wait_and_activate();

        /// True between a prefetch() and its matching wait_and_activate().
        bool prefetch_pending() const noexcept;

        /**
         * @brief Synchronous convenience: prefetch(block_ids) then wait_and_activate().
         *
         * Equivalent to the Phase 2 V1 behavior. Suitable when the trainer cannot
         * overlap a prefetch with the current iteration (e.g. first iteration,
         * or end-to-end debugging).
         */
        std::expected<void, std::string>
        load_and_activate(lfs::core::TieredCache& cache,
                          std::span<const std::size_t> block_ids);

        // === Active-buffer accessors (valid after a successful load_and_activate) ===

        /// Device pointer to the contiguous packed buffer of all resident blocks (raw bytes).
        const void* device_buffer() const noexcept;

        /// Mutable device pointer to the same buffer as @ref device_buffer.
        /// Used by writers (e.g. Tide's SOA→AOS repack after an optimizer step)
        /// that need to update the resident bytes in place before the next prefetch.
        /// Callers must NOT outlive the next @ref wait_and_activate() call.
        void* mutable_device_buffer() noexcept;

        /// Per-resident-block slice table, in the order the rasterizer should see them.
        std::span<const BlockSlice> active_slices() const noexcept;

        /// Number of resident blocks currently in VRAM.
        std::size_t active_block_count() const noexcept;

        /// Total Gaussian count across all resident blocks (= active_block_count * block_size,
        /// minus any short tail if the last global block is partial).
        std::size_t active_gaussian_count() const noexcept;

        const Config& config() const noexcept { return config_; }
        const Stats& stats() const noexcept { return stats_; }

    private:
        WorkingSet() = default;

        struct Impl;
        std::unique_ptr<Impl> impl_;

        Config config_{};
        Stats stats_{};
    };

} // namespace lfs::training::tide
