/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace lfs::core {

    /**
     * @brief Out-of-core block storage for Gaussian parameter tables.
     *
     * Implements the SSD-tier of the SSD-CPU-GPU hierarchy described in
     * TideGS (arXiv:2605.20150). The full Gaussian parameter table is stored
     * as a contiguous, page-aligned binary file ("base segment") and updates
     * are appended to log-structured "patch segments" without overwriting
     * the base.
     *
     * Layout (per-Gaussian, packed, host-endian fp32):
     *   [xyz(3) | scaling(3) | rotation(4) | opacity(1) | dc(3) | rest(45)]
     *   = 59 floats = 236 bytes per Gaussian.
     *
     * A block of B=4096 Gaussians is exactly 944 KiB = 236 contiguous 4-KiB pages.
     *
     * The store is **block-addressable**: callers read whole blocks via @ref read_block
     * and write whole blocks via @ref write_block. Per-block bounds (center+radius for
     * frustum culling) are stored in a sidecar file.
     *
     * Thread-safety: read_block / write_block / lookup are safe under concurrent calls.
     * Construction and @ref close are not.
     */
    class LFS_CORE_API BlockStore {
    public:
        // === Constants matching TideGS paper ===
        static constexpr std::size_t kAttributesPerGaussian = 59;          ///< xyz(3)+scale(3)+rot(4)+opacity(1)+dc(3)+rest(45)
        static constexpr std::size_t kBytesPerGaussian = kAttributesPerGaussian * sizeof(float); ///< 236 B
        static constexpr std::size_t kDefaultBlockSize = 4096;             ///< Gaussians per block (TideGS B)
        static constexpr std::size_t kPageSize = 4096;                     ///< OS page size for mmap alignment

        /**
         * @brief Coarse spatial bounds for a block — used for CPU-side frustum culling.
         *
         * The TideGS paper uses bounding spheres (Sec. 3.3, Fig. 3), which are
         * cheaper to test against frustum planes than AABBs (signed distance
         * to plane vs. radius). We mirror that choice.
         */
        struct BlockBounds {
            float center[3];
            float radius;
        };

        /**
         * @brief Per-block index entry (TideGS Eq. 3): (file_id, offset, size, version).
         *
         * file_id == 0 denotes the immutable base segment; later file_ids denote
         * patch segments. Reads always consult the latest version.
         */
        struct IndexEntry {
            std::uint32_t file_id; ///< 0 = base, 1+ = patch segments
            std::uint64_t offset;  ///< Byte offset within file_id
            std::uint32_t size;    ///< Bytes (= block_size * kBytesPerGaussian for full blocks)
            std::uint32_t version; ///< Monotonically incremented on each write
        };

        /// Configuration captured at @ref open / @ref create time.
        struct Config {
            std::size_t block_size = kDefaultBlockSize;          ///< Gaussians per block
            std::size_t patch_segment_capacity_bytes = 1ull << 30; ///< Roll over patch file every 1 GiB
            bool prefault_base = false;                          ///< MAP_POPULATE-equivalent on open (slow first call)
        };

        BlockStore();
        ~BlockStore();

        BlockStore(const BlockStore&) = delete;
        BlockStore& operator=(const BlockStore&) = delete;
        BlockStore(BlockStore&&) noexcept;
        BlockStore& operator=(BlockStore&&) noexcept;

        // === Lifecycle ===

        /**
         * @brief Create a new store from already-prepared base data.
         *
         * @param dir Directory to create the store in (must not already contain a store).
         * @param num_blocks Total block count (final = ceil(N/B)).
         * @param block_bounds Per-block bounds, must have @p num_blocks entries.
         * @param base_bytes Raw base-segment bytes; must be exactly
         *        num_blocks * block_size * kBytesPerGaussian (last block may be padded).
         * @param config Optional configuration (block_size etc.).
         */
        static std::expected<std::unique_ptr<BlockStore>, std::string> create(
            const std::filesystem::path& dir,
            std::size_t num_blocks,
            std::span<const BlockBounds> block_bounds,
            std::span<const std::byte> base_bytes,
            const Config& config = {});

        /**
         * @brief Open an existing store from disk.
         *
         * Memory-maps the base segment and any patch segments, replays the index file.
         */
        static std::expected<std::unique_ptr<BlockStore>, std::string> open(
            const std::filesystem::path& dir,
            const Config& config = {});

        /**
         * @brief Flush index and any open patch segment, then unmap.
         *
         * Idempotent. Called automatically by destructor.
         */
        std::expected<void, std::string> close();

        // === Block access ===

        /**
         * @brief Read a block by id into the caller-supplied buffer.
         *
         * Resolves Index[block_id] to the latest version and copies that block's
         * payload into @p dst. @p dst.size() must equal @ref bytes_per_block().
         */
        std::expected<void, std::string> read_block(std::size_t block_id, std::span<std::byte> dst) const;

        /**
         * @brief Append a new block version to the active patch segment.
         *
         * Writes are sequential and never overwrite. Updates Index[block_id] atomically.
         * Returns the new version number on success.
         */
        std::expected<std::uint32_t, std::string> write_block(std::size_t block_id, std::span<const std::byte> src);

        /// Get the current index entry (latest version) for @p block_id.
        std::expected<IndexEntry, std::string> lookup(std::size_t block_id) const;

        /// Get the bounding sphere for a block. Bounds are mutable across writes via @ref update_bounds.
        BlockBounds get_bounds(std::size_t block_id) const;

        /// Refresh the bounding sphere for a block (e.g. after centers move during training).
        void update_bounds(std::size_t block_id, const BlockBounds& bounds);

        // === Introspection ===

        std::size_t num_blocks() const noexcept { return num_blocks_; }
        std::size_t block_size() const noexcept { return config_.block_size; }
        std::size_t bytes_per_block() const noexcept { return config_.block_size * kBytesPerGaussian; }
        const std::filesystem::path& directory() const noexcept { return dir_; }

        /// Aggregate counters for telemetry parity with TideGS (cache_hits, etc.).
        struct Stats {
            std::atomic<std::uint64_t> reads{0};
            std::atomic<std::uint64_t> writes{0};
            std::atomic<std::uint64_t> bytes_read{0};
            std::atomic<std::uint64_t> bytes_written{0};
            std::atomic<std::uint64_t> patch_segments{0};
        };
        const Stats& stats() const noexcept { return *stats_; }

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;

        // Mirrored here for fast inline access.
        std::filesystem::path dir_;
        Config config_{};
        std::size_t num_blocks_ = 0;
        std::unique_ptr<Stats> stats_;

        // Bounds are updated frequently during training; protect with a coarse mutex
        // for now (per-block atomics can come later if profiling shows contention).
        mutable std::mutex bounds_mutex_;
        std::vector<BlockBounds> bounds_;
    };

} // namespace lfs::core
