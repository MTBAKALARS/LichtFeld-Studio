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
     * Sec. 3.5). A WorkingSet owns a single GPU buffer sized for @ref capacity_blocks
     * blocks of `BlockStore::bytes_per_block()` bytes each, plus a `block_to_local_slice`
     * mapping that lets the rasterizer find a specific block's contiguous slice in
     * the active buffer.
     *
     * **This is the Phase 2 V1: synchronous H2D loads.** The A/B double-buffer +
     * async prefetch pipeline (paper's Algorithm 1) is a deliberate follow-up;
     * a synchronous version is enough to wire Phase 3's trainer integration end-to-end
     * and validates the data plumbing on real data before we add overlap.
     *
     * Memory layout in the active buffer (CACHE order, identical to the BlockStore
     * on-disk format):
     *
     *     [block_0 (4096 Gaussians × 236 B)] [block_1 ...] ... [block_{N-1} ...]
     *
     * Per-Gaussian byte layout is fixed by @ref BlockStore::kBytesPerGaussian = 236.
     *
     * Thread-safety: not safe for concurrent calls. The trainer owns one WorkingSet
     * and drives it from the main training thread.
     */
    class WorkingSet {
    public:
        /// One contiguous slice of the active buffer that holds a particular block's bytes.
        struct BlockSlice {
            std::size_t block_id = 0;
            std::size_t local_index = 0; ///< 0..capacity_blocks-1; multiply by bytes_per_block for byte offset
        };

        struct Config {
            std::size_t capacity_blocks = 0;     ///< Hard cap on resident block count in VRAM
            std::size_t bytes_per_block = 0;     ///< Must match BlockStore::bytes_per_block()
            int cuda_device = 0;                 ///< GPU index for cudaSetDevice
        };

        struct Stats {
            std::uint64_t loads = 0;             ///< Number of begin_load → activate cycles
            std::uint64_t blocks_uploaded = 0;
            std::uint64_t bytes_uploaded = 0;
            std::uint64_t blocks_retained = 0;   ///< Blocks already resident from prior frame
        };

        ~WorkingSet();
        WorkingSet(const WorkingSet&) = delete;
        WorkingSet& operator=(const WorkingSet&) = delete;
        WorkingSet(WorkingSet&&) noexcept;
        WorkingSet& operator=(WorkingSet&&) noexcept;

        /**
         * @brief Allocate the GPU buffer and a dedicated CUDA stream.
         *
         * @return WorkingSet on success; descriptive error on cuMalloc/stream failure.
         */
        static std::expected<std::unique_ptr<WorkingSet>, std::string>
        create(const Config& config);

        /**
         * @brief Stage the next resident set into VRAM and activate it.
         *
         * Synchronously H2D-copies each non-retained block from the @p cache into
         * the GPU buffer, then `cudaStreamSynchronize` on the dedicated stream.
         * Blocks already resident from the prior call are kept in place
         * (their `local_index` is preserved); freshly evicted block slots are
         * reused for the new arrivals.
         *
         * @param cache Source of pinned-host block bytes. Each block_id is pinned
         *              via @ref TieredCache::get for the duration of the H2D, then unpinned.
         * @param block_ids Sorted, unique list of block_ids that must be resident
         *                  after this call returns. `block_ids.size() <= capacity_blocks`.
         */
        std::expected<void, std::string>
        load_and_activate(lfs::core::TieredCache& cache,
                          std::span<const std::size_t> block_ids);

        // === Active-buffer accessors (valid after a successful load_and_activate) ===

        /// Device pointer to the contiguous packed buffer of all resident blocks (raw bytes).
        const void* device_buffer() const noexcept;

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
