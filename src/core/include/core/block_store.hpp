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
        /// Per-Gaussian byte footprint of Adam optimizer moments (m + v, fp32).
        /// Stored in a separate optional region so the immutable base segment
        /// stays at 236 B/Gaussian (existing v1 stores reusable without re-bake).
        /// Layout matches @ref kAttributesPerGaussian one-to-one in SOA order, so a
        /// resident WorkingSet slot can address moments by the same param-offset
        /// table as parameters. See @ref Phase 3.5.3 design in TideGS paper Sec. 3.4.
        static constexpr std::size_t kAdamMomentsBytesPerGaussian =
            2u * kAttributesPerGaussian * sizeof(float); ///< 472 B (m,v fp32 × 59 scalars)
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
            /// When true at @ref create time, allocates a per-block Adam moments
            /// region of @ref kAdamMomentsBytesPerGaussian × block_size bytes per
            /// block in a separate `moments.bin` file, zero-initialized. Triggers a
            /// manifest v2 header. Default false keeps existing stores byte-for-byte
            /// compatible (manifest stays v1, no moments file written).
            /// On @ref open this field is ignored; the manifest dictates whether
            /// moments are present (queryable via @ref has_moments).
            bool with_moments = false;
        };

        BlockStore();
        ~BlockStore();

        BlockStore(const BlockStore&) = delete;
        BlockStore& operator=(const BlockStore&) = delete;
        // Non-movable: held via std::unique_ptr<BlockStore> from the static
        // factories (create / open / stream_ply_to_base). The class owns a
        // std::mutex directly (bounds_mutex_), which is neither copyable nor
        // movable, so the implicitly-synthesized move operations cannot exist.
        BlockStore(BlockStore&&)            = delete;
        BlockStore& operator=(BlockStore&&) = delete;

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
         * @brief CPU-side view of an in-memory Gaussian set for streaming initialization.
         *
         * All spans must contain @ref num_gaussians elements along the leading axis,
         * laid out contiguously in float32. Optimizer-raw values are expected (i.e.
         * pre-sigmoid opacity, log-space scaling) — these match SplatData's `*_raw`
         * accessors and the on-disk PLY convention used by 3DGS.
         *
         * Per-attribute layout:
         *   - means    : N * 3 (xyz)
         *   - scaling  : N * 3 (raw / log-space)
         *   - rotation : N * 4 (quaternion, layout matches SplatData)
         *   - opacity  : N * 1 (raw / pre-sigmoid)
         *   - sh0      : N * 3 (DC SH coefficients, R/G/B)
         *   - shN      : N * (sh_rest_components) — typically 45 for SH-3
         *
         * The store layout always packs CACHE order:
         *   [xyz | scale | rot | opacity | dc | rest=45]
         * If sh0/shN have fewer components than 3/45, the remainder is zero-padded;
         * if more, extras are truncated with a warning.
         */
        struct PlySource {
            std::span<const float> means;
            std::span<const float> scaling;
            std::span<const float> rotation;
            std::span<const float> opacity;
            std::span<const float> sh0;
            std::span<const float> shN;
            std::size_t num_gaussians = 0;
            std::size_t sh_rest_components = 45; ///< 15 * 3 for SH-3
        };

        /// Optional knobs for @ref stream_ply_to_base.
        struct StreamPlyOptions {
            /// Hard ceiling on N for the in-memory path. Above this we refuse to
            /// allocate the output buffer (would exceed practical RAM); a future
            /// external-memory bucket sort path will lift this.
            std::size_t max_in_memory_gaussians = 250'000'000ULL;
            /// Number of Morton bits per axis (3 * bits ≤ 64). 21 is the standard
            /// choice and yields a uniform 2 097 152^3 grid resolution.
            std::uint32_t morton_bits_per_axis = 21;
        };

        /**
         * @brief Build a fresh store from a CPU-side Gaussian set.
         *
         * Implements TideGS's "streaming PLY init" (Sec. 3.2) but currently fully
         * in memory: the caller supplies CPU-resident spans, this function computes
         * a 21-bit-per-axis Morton ordering over the global bounding box, reorders
         * attributes into CACHE layout, derives per-block bounding spheres, and
         * delegates to @ref create to materialize the on-disk store.
         *
         * Caps at @ref StreamPlyOptions::max_in_memory_gaussians; an external-memory
         * bucket-spill variant for billion-scale PLYs is tracked as future work.
         *
         * @param dir Target directory (must not already contain a store).
         * @param src CPU-side Gaussian arrays (see @ref PlySource).
         * @param config Same Config that will be used to subsequently @ref open the store.
         * @param opts Streaming-init knobs; defaults are reasonable.
         */
        static std::expected<std::unique_ptr<BlockStore>, std::string> stream_ply_to_base(
            const std::filesystem::path& dir,
            const PlySource& src,
            const Config& config = {},
            const StreamPlyOptions& opts = {});

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

        // === Adam moments (optional sidecar region) ===

        /// True if this store has a moments region (was created with
        /// @ref Config::with_moments == true, or opened from a manifest-v2 store
        /// whose moments_bytes_per_block > 0).
        bool has_moments() const noexcept { return moments_bytes_per_block_ != 0; }

        /// Per-block byte count of the Adam moments region, or 0 if absent.
        /// Always either 0 or `block_size * kAdamMomentsBytesPerGaussian`.
        std::size_t moments_bytes_per_block() const noexcept { return moments_bytes_per_block_; }

        /// Read a block's Adam moments into @p dst.
        /// @p dst.size() must equal @ref moments_bytes_per_block(). Errors if the
        /// store has no moments region.
        std::expected<void, std::string> read_moments(std::size_t block_id, std::span<std::byte> dst) const;

        /// Overwrite a block's Adam moments in place. Unlike @ref write_block, moments
        /// are not versioned (Adam state is transient training state) — we overwrite
        /// the per-block slot directly. @p src.size() must equal @ref moments_bytes_per_block().
        std::expected<void, std::string> write_moments(std::size_t block_id, std::span<const std::byte> src);

        /// Get the bounding sphere for a block. Bounds are mutable across writes via @ref update_bounds.
        BlockBounds get_bounds(std::size_t block_id) const;

        /// Bulk snapshot all per-block bounds under a single mutex acquisition.
        /// Resizes @p out to `num_blocks()` and copies the current bounds vector.
        /// Used by Tide pre_forward to feed the frustum culler once per iteration
        /// without paying num_blocks() lock acquisitions.
        void snapshot_bounds(std::vector<BlockBounds>& out) const;

        /// Refresh the bounding sphere for a block (e.g. after centers move during training).
        void update_bounds(std::size_t block_id, const BlockBounds& bounds);

        // === Introspection ===

        std::size_t num_blocks() const noexcept { return num_blocks_; }
        std::size_t block_size() const noexcept { return config_.block_size; }
        std::size_t bytes_per_block() const noexcept { return config_.block_size * kBytesPerGaussian; }
        const std::filesystem::path& directory() const noexcept { return dir_; }

        /// Manifest schema version this store was opened with (1 = legacy, 2 = with optional moments).
        std::uint32_t manifest_version() const noexcept { return manifest_version_; }

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

        // Optional Adam moments sidecar region. 0 if absent.
        std::size_t moments_bytes_per_block_ = 0;
        // Manifest schema version (1 or 2). Used by tests and diagnostics.
        std::uint32_t manifest_version_ = 0;
    };

} // namespace lfs::core
