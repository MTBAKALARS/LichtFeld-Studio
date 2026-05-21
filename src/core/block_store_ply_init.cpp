/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file block_store_ply_init.cpp
 * @brief In-memory streaming-init factory: PLY-shaped CPU spans → BlockStore.
 *
 * Ports the data-layout half of TideGS's `storage/streaming_ply_init.py`
 * (arXiv:2605.20150). The paper performs the same Morton-order + per-block
 * bounding-sphere preparation, but spills to bucketed scratch files so that
 * a billion-Gaussian PLY can be processed with a 16 GiB RAM budget. Our
 * RTX 4090 / 128 GB workstation tolerates a fully in-memory implementation up
 * to ~250M Gaussians (≈59 GiB output buffer); the external-memory variant is
 * tracked as future work in the Phase 1 design doc.
 *
 * Inputs are CPU spans with the SplatData/PLY convention (raw opacity, raw
 * log-scaling). The output base segment uses the CACHE layout that the rest
 * of the BlockStore — and the upcoming GPUWorkingSet — expects:
 *
 *     [xyz(3) | scaling(3) | rotation(4) | opacity(1) | dc(3) | rest(45)]
 *
 * No CUDA, TBB, or filesystem APIs beyond what `block_store.cpp` already uses.
 */

#include "core/block_store.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

namespace lfs::core {

    namespace {

        // === Morton encoding (21 bits per axis, packed into uint64) ===========

        /// Spread a 21-bit integer across every third bit of a 64-bit word.
        /// Standard "magic bits" sequence; correctness verified against
        /// libmorton's BMI2-free reference implementation.
        inline std::uint64_t expand_bits_21(std::uint32_t v) noexcept {
            std::uint64_t x = v & 0x1FFFFFu; // mask to 21 bits
            x = (x | (x << 32)) & 0x001F00000000FFFFull;
            x = (x | (x << 16)) & 0x001F0000FF0000FFull;
            x = (x | (x << 8)) & 0x100F00F00F00F00Full;
            x = (x | (x << 4)) & 0x10C30C30C30C30C3ull;
            x = (x | (x << 2)) & 0x1249249249249249ull;
            return x;
        }

        inline std::uint64_t morton_encode_3d(std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
            return (expand_bits_21(z) << 2) | (expand_bits_21(y) << 1) | expand_bits_21(x);
        }

        // === Bounding box / quantization helpers =============================

        struct AABB {
            float min[3]{+std::numeric_limits<float>::infinity(),
                         +std::numeric_limits<float>::infinity(),
                         +std::numeric_limits<float>::infinity()};
            float max[3]{-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()};
        };

        AABB compute_aabb(std::span<const float> xyz, std::size_t n) {
            AABB box;
            for (std::size_t i = 0; i < n; ++i) {
                const float x = xyz[3 * i + 0];
                const float y = xyz[3 * i + 1];
                const float z = xyz[3 * i + 2];
                if (x < box.min[0]) box.min[0] = x;
                if (y < box.min[1]) box.min[1] = y;
                if (z < box.min[2]) box.min[2] = z;
                if (x > box.max[0]) box.max[0] = x;
                if (y > box.max[1]) box.max[1] = y;
                if (z > box.max[2]) box.max[2] = z;
            }
            return box;
        }

        BlockStore::BlockBounds compute_block_sphere(const float* xyz_packed, std::size_t count) noexcept {
            // Pass 1: centroid.
            double cx = 0.0, cy = 0.0, cz = 0.0;
            for (std::size_t i = 0; i < count; ++i) {
                cx += xyz_packed[BlockStore::kAttributesPerGaussian * i + 0];
                cy += xyz_packed[BlockStore::kAttributesPerGaussian * i + 1];
                cz += xyz_packed[BlockStore::kAttributesPerGaussian * i + 2];
            }
            const double inv = (count == 0) ? 0.0 : 1.0 / static_cast<double>(count);
            BlockStore::BlockBounds b{};
            b.center[0] = static_cast<float>(cx * inv);
            b.center[1] = static_cast<float>(cy * inv);
            b.center[2] = static_cast<float>(cz * inv);
            // Pass 2: max squared distance.
            float r2 = 0.0f;
            for (std::size_t i = 0; i < count; ++i) {
                const float dx = xyz_packed[BlockStore::kAttributesPerGaussian * i + 0] - b.center[0];
                const float dy = xyz_packed[BlockStore::kAttributesPerGaussian * i + 1] - b.center[1];
                const float dz = xyz_packed[BlockStore::kAttributesPerGaussian * i + 2] - b.center[2];
                const float d2 = dx * dx + dy * dy + dz * dz;
                if (d2 > r2) r2 = d2;
            }
            b.radius = std::sqrt(r2);
            return b;
        }

    } // namespace

    std::expected<std::unique_ptr<BlockStore>, std::string>
    BlockStore::stream_ply_to_base(const std::filesystem::path& dir,
                                   const PlySource& src,
                                   const Config& config,
                                   const StreamPlyOptions& opts) {
        const std::size_t N = src.num_gaussians;
        if (N == 0) {
            return std::unexpected<std::string>("stream_ply_to_base: num_gaussians is zero");
        }
        if (N > opts.max_in_memory_gaussians) {
            return std::unexpected<std::string>(
                "stream_ply_to_base: " + std::to_string(N) +
                " gaussians exceeds in-memory cap of " +
                std::to_string(opts.max_in_memory_gaussians) +
                "; external-memory bucket-spill path is not yet implemented");
        }
        if (opts.morton_bits_per_axis == 0 || opts.morton_bits_per_axis > 21) {
            return std::unexpected<std::string>(
                "stream_ply_to_base: morton_bits_per_axis must be in [1, 21]");
        }

        // === Validate input span sizes =====================================
        auto require = [](std::span<const float> s, std::size_t expected, const char* name)
            -> std::expected<void, std::string> {
            if (s.size() != expected) {
                return std::unexpected<std::string>(
                    std::string("stream_ply_to_base: ") + name + " span size " +
                    std::to_string(s.size()) + " != expected " + std::to_string(expected));
            }
            return {};
        };

        if (auto r = require(src.means, 3 * N, "means"); !r) return std::unexpected(r.error());
        if (auto r = require(src.scaling, 3 * N, "scaling"); !r) return std::unexpected(r.error());
        if (auto r = require(src.rotation, 4 * N, "rotation"); !r) return std::unexpected(r.error());
        if (auto r = require(src.opacity, 1 * N, "opacity"); !r) return std::unexpected(r.error());
        if (auto r = require(src.sh0, 3 * N, "sh0"); !r) return std::unexpected(r.error());
        if (src.shN.size() != src.sh_rest_components * N) {
            return std::unexpected<std::string>(
                "stream_ply_to_base: shN span size " + std::to_string(src.shN.size()) +
                " != sh_rest_components(" + std::to_string(src.sh_rest_components) +
                ") * N(" + std::to_string(N) + ")");
        }

        constexpr std::size_t kRestSlots = 45; // SH-3 rest layout in CACHE block
        const std::size_t copy_rest = std::min(src.sh_rest_components, kRestSlots);
        if (src.sh_rest_components > kRestSlots) {
            LOG_WARN("stream_ply_to_base: sh_rest_components={} > {}; truncating extra coefficients",
                     src.sh_rest_components, kRestSlots);
        }

        // === 1. Global bbox + Morton quantization grid =====================
        const AABB box = compute_aabb(src.means, N);
        // Guard against degenerate axes (single point, plane, etc.) — clamp to >0
        // so the quantization step never divides by zero.
        const double eps = 1e-12;
        const double extent[3] = {
            std::max<double>(static_cast<double>(box.max[0]) - box.min[0], eps),
            std::max<double>(static_cast<double>(box.max[1]) - box.min[1], eps),
            std::max<double>(static_cast<double>(box.max[2]) - box.min[2], eps),
        };
        const std::uint32_t grid_max = (opts.morton_bits_per_axis >= 32)
                                           ? 0xFFFFFFFFu
                                           : ((1u << opts.morton_bits_per_axis) - 1u);
        const double scale[3] = {
            static_cast<double>(grid_max) / extent[0],
            static_cast<double>(grid_max) / extent[1],
            static_cast<double>(grid_max) / extent[2],
        };

        // === 2. Per-Gaussian Morton codes ==================================
        std::vector<std::uint64_t> codes;
        try {
            codes.assign(N, 0);
        } catch (const std::bad_alloc&) {
            return std::unexpected<std::string>(
                "stream_ply_to_base: out of memory allocating Morton codes (" +
                std::to_string(N * sizeof(std::uint64_t)) + " bytes)");
        }

        for (std::size_t i = 0; i < N; ++i) {
            const double nx = (static_cast<double>(src.means[3 * i + 0]) - box.min[0]) * scale[0];
            const double ny = (static_cast<double>(src.means[3 * i + 1]) - box.min[1]) * scale[1];
            const double nz = (static_cast<double>(src.means[3 * i + 2]) - box.min[2]) * scale[2];
            const std::uint32_t qx = static_cast<std::uint32_t>(
                std::clamp(nx, 0.0, static_cast<double>(grid_max)));
            const std::uint32_t qy = static_cast<std::uint32_t>(
                std::clamp(ny, 0.0, static_cast<double>(grid_max)));
            const std::uint32_t qz = static_cast<std::uint32_t>(
                std::clamp(nz, 0.0, static_cast<double>(grid_max)));
            codes[i] = morton_encode_3d(qx, qy, qz);
        }

        // === 3. Sort permutation by Morton code ============================
        std::vector<std::uint32_t> perm;
        try {
            perm.resize(N);
        } catch (const std::bad_alloc&) {
            return std::unexpected<std::string>(
                "stream_ply_to_base: out of memory allocating permutation");
        }
        std::iota(perm.begin(), perm.end(), 0u);
        // N must fit in uint32_t for our perm encoding; otherwise widen.
        if (N > std::numeric_limits<std::uint32_t>::max()) {
            return std::unexpected<std::string>(
                "stream_ply_to_base: N exceeds 2^32; uint32 perm cannot index it");
        }
        std::sort(perm.begin(), perm.end(),
                  [&codes](std::uint32_t a, std::uint32_t b) {
                      return codes[a] < codes[b];
                  });
        // Codes no longer needed; reclaim ~8 GiB at N=1B. Negligible at our scale
        // but cheap insurance for the hot 250M ceiling.
        codes.clear();
        codes.shrink_to_fit();

        // === 4. Allocate base buffer (block-padded) ========================
        const std::size_t B = config.block_size == 0 ? kDefaultBlockSize : config.block_size;
        const std::size_t num_blocks = (N + B - 1) / B;
        const std::size_t padded_N = num_blocks * B;
        const std::size_t base_floats = padded_N * kAttributesPerGaussian;
        const std::size_t base_bytes_size = base_floats * sizeof(float);

        std::vector<float> base;
        try {
            base.assign(base_floats, 0.0f);
        } catch (const std::bad_alloc&) {
            return std::unexpected<std::string>(
                "stream_ply_to_base: out of memory allocating base buffer (" +
                std::to_string(base_bytes_size) + " bytes); reduce N or wait for "
                "external-memory variant");
        }

        // === 5. Apply permutation, reorder into CACHE layout ===============
        // CACHE per-Gaussian: [xyz(3) | scale(3) | rot(4) | opacity(1) | dc(3) | rest(45)]
        constexpr std::size_t off_xyz = 0;
        constexpr std::size_t off_scl = 3;
        constexpr std::size_t off_rot = 6;
        constexpr std::size_t off_opa = 10;
        constexpr std::size_t off_dc = 11;
        constexpr std::size_t off_rest = 14;
        static_assert(off_rest + kRestSlots == kAttributesPerGaussian,
                      "CACHE attribute offsets do not sum to kAttributesPerGaussian");

        for (std::size_t out_i = 0; out_i < N; ++out_i) {
            const std::size_t src_i = perm[out_i];
            float* dst = base.data() + out_i * kAttributesPerGaussian;

            dst[off_xyz + 0] = src.means[3 * src_i + 0];
            dst[off_xyz + 1] = src.means[3 * src_i + 1];
            dst[off_xyz + 2] = src.means[3 * src_i + 2];

            dst[off_scl + 0] = src.scaling[3 * src_i + 0];
            dst[off_scl + 1] = src.scaling[3 * src_i + 1];
            dst[off_scl + 2] = src.scaling[3 * src_i + 2];

            dst[off_rot + 0] = src.rotation[4 * src_i + 0];
            dst[off_rot + 1] = src.rotation[4 * src_i + 1];
            dst[off_rot + 2] = src.rotation[4 * src_i + 2];
            dst[off_rot + 3] = src.rotation[4 * src_i + 3];

            dst[off_opa] = src.opacity[src_i];

            dst[off_dc + 0] = src.sh0[3 * src_i + 0];
            dst[off_dc + 1] = src.sh0[3 * src_i + 1];
            dst[off_dc + 2] = src.sh0[3 * src_i + 2];

            const float* rest_src = src.shN.data() + src.sh_rest_components * src_i;
            std::memcpy(dst + off_rest, rest_src, copy_rest * sizeof(float));
            // Trailing slots already zero-initialized in base.assign().
        }

        // Padding tail (out_i in [N, padded_N)): leave zero-initialized.
        // A zero quaternion (0,0,0,0) is degenerate but never read because
        // the IndexEntry size for the tail block reports the true byte count
        // and downstream consumers honor it. (We still write a full block's
        // worth of bytes to keep base.bin aligned.)

        // Free perm before computing bounds (saves 4 GiB at N=1B).
        perm.clear();
        perm.shrink_to_fit();

        // === 6. Per-block bounding spheres =================================
        std::vector<BlockBounds> bounds;
        try {
            bounds.resize(num_blocks);
        } catch (const std::bad_alloc&) {
            return std::unexpected<std::string>(
                "stream_ply_to_base: out of memory allocating bounds vector");
        }
        for (std::size_t b = 0; b < num_blocks; ++b) {
            const std::size_t first = b * B;
            const std::size_t end = std::min(first + B, N);
            const std::size_t count = end - first;
            const float* block_ptr = base.data() + first * kAttributesPerGaussian;
            bounds[b] = compute_block_sphere(block_ptr, count);
        }

        // === 7. Hand off to existing on-disk create() ======================
        std::span<const std::byte> base_view{
            reinterpret_cast<const std::byte*>(base.data()), base_bytes_size};
        std::span<const BlockBounds> bounds_view{bounds};

        LOG_INFO("stream_ply_to_base: N={} num_blocks={} base_bytes={} (~{:.2f} GiB)",
                 N, num_blocks, base_bytes_size,
                 static_cast<double>(base_bytes_size) / (1ULL << 30));

        return BlockStore::create(dir, num_blocks, bounds_view, base_view, config);
    }

} // namespace lfs::core
