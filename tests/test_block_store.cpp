/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_block_store.cpp
 * @brief Unit tests for BlockStore + TieredCache + stream_ply_to_base.
 *
 * Tests are CPU-only (no CUDA) and operate on small synthetic Gaussian sets so
 * the suite finishes in well under a second. We verify:
 *   - Round-trip read/write of base blocks via mmap + patch segments
 *   - Index version monotonicity on overwrite
 *   - stream_ply_to_base reorders attributes into CACHE layout
 *   - Bounding spheres enclose the Gaussians in their block
 *   - TieredCache hit/miss accounting and dirty-flush completes
 *   - Span-size validation in stream_ply_to_base rejects malformed input
 */

#include "core/block_store.hpp"
#include "core/tiered_cache.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <span>
#include <vector>

namespace {

    namespace fs = std::filesystem;
    using lfs::core::BlockStore;
    using lfs::core::TieredCache;

    // RAII helper: makes a unique temp dir and removes it on scope exit.
    class TempDir {
    public:
        TempDir() {
            std::random_device rd;
            const auto name = "lfs_block_store_test_" + std::to_string(rd()) + "_" +
                              std::to_string(rd());
            path_ = fs::temp_directory_path() / name;
            fs::create_directories(path_);
        }
        ~TempDir() {
            std::error_code ec;
            fs::remove_all(path_, ec);
        }
        TempDir(const TempDir&) = delete;
        TempDir& operator=(const TempDir&) = delete;
        const fs::path& path() const { return path_; }

    private:
        fs::path path_;
    };

    // Synthetic Gaussians on a regular grid — easy to verify Morton ordering
    // because positions are deterministic.
    struct SyntheticData {
        std::vector<float> means;
        std::vector<float> scaling;
        std::vector<float> rotation;
        std::vector<float> opacity;
        std::vector<float> sh0;
        std::vector<float> shN;
        std::size_t n = 0;
        static constexpr std::size_t kRest = 45;
    };

    SyntheticData make_grid(std::size_t side) {
        SyntheticData d;
        d.n = side * side * side;
        d.means.reserve(3 * d.n);
        d.scaling.assign(3 * d.n, 0.0f);
        d.rotation.assign(4 * d.n, 0.0f);
        d.opacity.assign(d.n, 0.0f);
        d.sh0.assign(3 * d.n, 0.0f);
        d.shN.assign(SyntheticData::kRest * d.n, 0.0f);

        const float step = 1.0f;
        std::size_t idx = 0;
        for (std::size_t z = 0; z < side; ++z) {
            for (std::size_t y = 0; y < side; ++y) {
                for (std::size_t x = 0; x < side; ++x, ++idx) {
                    d.means.push_back(static_cast<float>(x) * step);
                    d.means.push_back(static_cast<float>(y) * step);
                    d.means.push_back(static_cast<float>(z) * step);
                    // Encode the source index into opacity so we can recover
                    // the Morton permutation after stream_ply_to_base.
                    d.opacity[idx] = static_cast<float>(idx);
                    // Identity quaternion (W = 1); LichtFeld layout is (w,x,y,z) or
                    // (x,y,z,w) — irrelevant here, we only verify byte-equal copy.
                    d.rotation[4 * idx + 0] = 1.0f;
                    // Distinct scaling per axis to detect any axis swap.
                    d.scaling[3 * idx + 0] = -1.0f;
                    d.scaling[3 * idx + 1] = -2.0f;
                    d.scaling[3 * idx + 2] = -3.0f;
                    // Distinct sh0 / shN per Gaussian.
                    d.sh0[3 * idx + 0] = static_cast<float>(idx) + 0.1f;
                    d.sh0[3 * idx + 1] = static_cast<float>(idx) + 0.2f;
                    d.sh0[3 * idx + 2] = static_cast<float>(idx) + 0.3f;
                    d.shN[SyntheticData::kRest * idx + 0] = static_cast<float>(idx) + 0.5f;
                    d.shN[SyntheticData::kRest * idx + (SyntheticData::kRest - 1)] =
                        static_cast<float>(idx) + 0.6f;
                }
            }
        }
        return d;
    }

    BlockStore::PlySource as_source(const SyntheticData& d) {
        BlockStore::PlySource s;
        s.means = std::span<const float>(d.means);
        s.scaling = std::span<const float>(d.scaling);
        s.rotation = std::span<const float>(d.rotation);
        s.opacity = std::span<const float>(d.opacity);
        s.sh0 = std::span<const float>(d.sh0);
        s.shN = std::span<const float>(d.shN);
        s.num_gaussians = d.n;
        s.sh_rest_components = SyntheticData::kRest;
        return s;
    }

} // namespace

// ============================================================================
// stream_ply_to_base: validation
// ============================================================================

TEST(BlockStoreStreamPlyTest, RejectsZeroGaussians) {
    TempDir tmp;
    BlockStore::PlySource src{};
    src.num_gaussians = 0;
    auto r = BlockStore::stream_ply_to_base(tmp.path() / "store", src);
    ASSERT_FALSE(r.has_value());
    EXPECT_NE(r.error().find("zero"), std::string::npos);
}

TEST(BlockStoreStreamPlyTest, RejectsMismatchedMeansSpan) {
    TempDir tmp;
    SyntheticData d = make_grid(4);
    d.means.pop_back(); // 3*N - 1
    auto src = as_source(d);
    auto r = BlockStore::stream_ply_to_base(tmp.path() / "store", src);
    ASSERT_FALSE(r.has_value());
    EXPECT_NE(r.error().find("means"), std::string::npos);
}

TEST(BlockStoreStreamPlyTest, RejectsExceedingInMemoryCap) {
    TempDir tmp;
    BlockStore::PlySource src{};
    src.num_gaussians = 300'000'000ULL;
    BlockStore::StreamPlyOptions opts;
    opts.max_in_memory_gaussians = 250'000'000ULL;
    auto r = BlockStore::stream_ply_to_base(tmp.path() / "store", src, {}, opts);
    ASSERT_FALSE(r.has_value());
    EXPECT_NE(r.error().find("exceeds"), std::string::npos);
}

// ============================================================================
// stream_ply_to_base: round-trip
// ============================================================================

TEST(BlockStoreStreamPlyTest, RoundTripPreservesAllAttributes) {
    TempDir tmp;
    constexpr std::size_t side = 16; // 4096 Gaussians = exactly one block
    SyntheticData d = make_grid(side);
    ASSERT_EQ(d.n, BlockStore::kDefaultBlockSize);

    auto src = as_source(d);
    auto r = BlockStore::stream_ply_to_base(tmp.path() / "store", src);
    ASSERT_TRUE(r.has_value()) << r.error();
    auto store = std::move(r.value());

    ASSERT_EQ(store->num_blocks(), 1u);
    ASSERT_EQ(store->block_size(), BlockStore::kDefaultBlockSize);
    ASSERT_EQ(store->bytes_per_block(),
              BlockStore::kDefaultBlockSize * BlockStore::kBytesPerGaussian);

    std::vector<std::byte> buf(store->bytes_per_block());
    auto read_r = store->read_block(0, std::span<std::byte>(buf));
    ASSERT_TRUE(read_r.has_value()) << read_r.error();

    // Each Gaussian record (CACHE layout) is kAttributesPerGaussian floats:
    // [xyz | scale | rot | opacity | dc | rest=45]. We use opacity as the
    // source-index witness (encoded in make_grid).
    const auto* floats = reinterpret_cast<const float*>(buf.data());
    constexpr std::size_t A = BlockStore::kAttributesPerGaussian;

    std::vector<int> seen(d.n, 0);
    for (std::size_t i = 0; i < d.n; ++i) {
        const float* g = floats + i * A;
        const std::size_t src_idx = static_cast<std::size_t>(g[10]); // opacity slot
        ASSERT_LT(src_idx, d.n);
        seen[src_idx]++;

        // xyz must equal the source means at src_idx
        EXPECT_FLOAT_EQ(g[0], d.means[3 * src_idx + 0]);
        EXPECT_FLOAT_EQ(g[1], d.means[3 * src_idx + 1]);
        EXPECT_FLOAT_EQ(g[2], d.means[3 * src_idx + 2]);
        // scale (3)
        EXPECT_FLOAT_EQ(g[3], -1.0f);
        EXPECT_FLOAT_EQ(g[4], -2.0f);
        EXPECT_FLOAT_EQ(g[5], -3.0f);
        // rotation (4) — identity quaternion
        EXPECT_FLOAT_EQ(g[6], 1.0f);
        EXPECT_FLOAT_EQ(g[7], 0.0f);
        EXPECT_FLOAT_EQ(g[8], 0.0f);
        EXPECT_FLOAT_EQ(g[9], 0.0f);
        // dc (3) at offset 11..13
        EXPECT_FLOAT_EQ(g[11], static_cast<float>(src_idx) + 0.1f);
        EXPECT_FLOAT_EQ(g[12], static_cast<float>(src_idx) + 0.2f);
        EXPECT_FLOAT_EQ(g[13], static_cast<float>(src_idx) + 0.3f);
        // rest (45) at offset 14..58 — we set first and last entries
        EXPECT_FLOAT_EQ(g[14], static_cast<float>(src_idx) + 0.5f);
        EXPECT_FLOAT_EQ(g[14 + 44], static_cast<float>(src_idx) + 0.6f);
    }

    // Permutation must be a bijection.
    for (std::size_t i = 0; i < d.n; ++i) {
        EXPECT_EQ(seen[i], 1) << "src_idx " << i << " seen " << seen[i] << " times";
    }

    // Bounds sphere should enclose every Gaussian in the block.
    const auto bounds = store->get_bounds(0);
    for (std::size_t i = 0; i < d.n; ++i) {
        const float dx = d.means[3 * i + 0] - bounds.center[0];
        const float dy = d.means[3 * i + 1] - bounds.center[1];
        const float dz = d.means[3 * i + 2] - bounds.center[2];
        const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        EXPECT_LE(dist, bounds.radius + 1e-4f)
            << "Gaussian " << i << " outside sphere by " << (dist - bounds.radius);
    }
}

TEST(BlockStoreStreamPlyTest, MultiBlockTailIsPadded) {
    TempDir tmp;
    // Just over one block so num_blocks=2 and second block is mostly padding.
    SyntheticData d = make_grid(17); // 17^3 = 4913, > 4096
    auto src = as_source(d);
    auto r = BlockStore::stream_ply_to_base(tmp.path() / "store", src);
    ASSERT_TRUE(r.has_value()) << r.error();
    auto store = std::move(r.value());
    EXPECT_EQ(store->num_blocks(), 2u);

    // Both blocks must be readable at full bytes_per_block.
    std::vector<std::byte> buf(store->bytes_per_block());
    EXPECT_TRUE(store->read_block(0, std::span<std::byte>(buf)).has_value());
    EXPECT_TRUE(store->read_block(1, std::span<std::byte>(buf)).has_value());
}

// ============================================================================
// BlockStore: patch writes
// ============================================================================

TEST(BlockStoreTest, WriteBlockBumpsVersion) {
    TempDir tmp;
    SyntheticData d = make_grid(16);
    auto r = BlockStore::stream_ply_to_base(tmp.path() / "store", as_source(d));
    ASSERT_TRUE(r.has_value()) << r.error();
    auto store = std::move(r.value());

    auto entry0 = store->lookup(0);
    ASSERT_TRUE(entry0.has_value());
    EXPECT_EQ(entry0.value().file_id, 0u); // base
    EXPECT_EQ(entry0.value().version, 0u);

    std::vector<std::byte> payload(store->bytes_per_block(), std::byte{0xAB});
    auto v1 = store->write_block(0, std::span<const std::byte>(payload));
    ASSERT_TRUE(v1.has_value()) << v1.error();
    EXPECT_GT(v1.value(), 0u);

    auto entry1 = store->lookup(0);
    ASSERT_TRUE(entry1.has_value());
    EXPECT_GE(entry1.value().file_id, 1u); // patch segment
    EXPECT_EQ(entry1.value().version, v1.value());

    std::vector<std::byte> readback(store->bytes_per_block());
    ASSERT_TRUE(store->read_block(0, std::span<std::byte>(readback)).has_value());
    EXPECT_EQ(std::memcmp(readback.data(), payload.data(), payload.size()), 0);
}

// ============================================================================
// TieredCache
// ============================================================================

// Adopt a unique_ptr<BlockStore> into a shared_ptr so TieredCache (which takes
// shared ownership of the store) can keep it alive for its full lifetime.
static std::shared_ptr<BlockStore> share(std::unique_ptr<BlockStore> p) {
    return std::shared_ptr<BlockStore>(std::move(p));
}

TEST(TieredCacheTest, HitMissAccounting) {
    TempDir tmp;
    SyntheticData d = make_grid(16);
    auto r = BlockStore::stream_ply_to_base(tmp.path() / "store", as_source(d));
    ASSERT_TRUE(r.has_value()) << r.error();
    auto store = share(std::move(r.value()));

    TieredCache::Config cfg;
    cfg.capacity_blocks = 4;
    TieredCache cache(store, cfg);

    // First get → miss
    auto v1 = cache.get(0);
    ASSERT_TRUE(v1.has_value()) << v1.error();
    EXPECT_EQ(v1.value().size(), store->bytes_per_block());
    cache.unpin(0, /*dirty=*/false);

    // Second get → hit
    auto v2 = cache.get(0);
    ASSERT_TRUE(v2.has_value()) << v2.error();
    cache.unpin(0, /*dirty=*/false);

    EXPECT_GE(cache.stats().hits.load(), 1u);
    EXPECT_GE(cache.stats().misses.load(), 1u);
}

TEST(TieredCacheTest, DirtyFlushReachesStore) {
    TempDir tmp;
    SyntheticData d = make_grid(16);
    auto r = BlockStore::stream_ply_to_base(tmp.path() / "store", as_source(d));
    ASSERT_TRUE(r.has_value()) << r.error();
    auto store = share(std::move(r.value()));

    TieredCache::Config cfg;
    cfg.capacity_blocks = 4;
    TieredCache cache(store, cfg);

    auto write_view = cache.pin_for_write(0);
    ASSERT_TRUE(write_view.has_value()) << write_view.error();
    std::fill(write_view.value().begin(), write_view.value().end(), std::byte{0x5A});
    cache.unpin(0, /*dirty=*/true);

    // Synchronous drain.
    auto flush_r = cache.flush_dirty();
    ASSERT_TRUE(flush_r.has_value()) << flush_r.error();

    // Confirm via direct store read.
    std::vector<std::byte> buf(store->bytes_per_block());
    ASSERT_TRUE(store->read_block(0, std::span<std::byte>(buf)).has_value());
    for (std::size_t i = 0; i < buf.size(); ++i) {
        ASSERT_EQ(buf[i], std::byte{0x5A}) << "byte " << i << " not flushed";
    }
}

// ============================================================================
// Phase 3.5.3a: Adam moments sidecar region (manifest v2)
// ============================================================================

namespace {
    // Build a small store directly via create() to control Config.with_moments.
    std::unique_ptr<BlockStore> make_store_with_moments(const fs::path& dir,
                                                       std::size_t num_blocks,
                                                       bool with_moments) {
        BlockStore::Config cfg;
        cfg.block_size = BlockStore::kDefaultBlockSize;
        cfg.with_moments = with_moments;
        std::vector<BlockStore::BlockBounds> bounds(num_blocks);
        for (std::size_t b = 0; b < num_blocks; ++b) {
            bounds[b].center[0] = static_cast<float>(b);
            bounds[b].center[1] = 0.0f;
            bounds[b].center[2] = 0.0f;
            bounds[b].radius = 1.0f;
        }
        const std::size_t bytes_per_block = cfg.block_size * BlockStore::kBytesPerGaussian;
        std::vector<std::byte> base(num_blocks * bytes_per_block, std::byte{0});
        for (std::size_t b = 0; b < num_blocks; ++b) {
            base[b * bytes_per_block] = static_cast<std::byte>(0xA0 + b);
        }
        auto r = BlockStore::create(dir, num_blocks,
                                    std::span<const BlockStore::BlockBounds>(bounds),
                                    std::span<const std::byte>(base),
                                    cfg);
        if (!r) return nullptr;
        return std::move(*r);
    }
} // namespace

TEST(BlockStoreMomentsTest, CreateWithoutMomentsHasNoMomentsRegion) {
    TempDir tmp;
    auto store = make_store_with_moments(tmp.path() / "store", 2, false);
    ASSERT_NE(store, nullptr);
    EXPECT_FALSE(store->has_moments());
    EXPECT_EQ(store->moments_bytes_per_block(), 0u);
    EXPECT_FALSE(fs::exists(tmp.path() / "store" / "moments.bin"));
    EXPECT_EQ(store->manifest_version(), 2u);

    std::vector<std::byte> dummy(1);
    EXPECT_FALSE(store->read_moments(0, std::span<std::byte>(dummy)).has_value());
    EXPECT_FALSE(store->write_moments(0, std::span<const std::byte>(dummy)).has_value());
}

TEST(BlockStoreMomentsTest, CreateWithMomentsAllocatesZeroedRegion) {
    TempDir tmp;
    constexpr std::size_t kNumBlocks = 3;
    auto store = make_store_with_moments(tmp.path() / "store", kNumBlocks, true);
    ASSERT_NE(store, nullptr);
    EXPECT_TRUE(store->has_moments());
    const std::size_t expected_per_block =
        BlockStore::kDefaultBlockSize * BlockStore::kAdamMomentsBytesPerGaussian;
    EXPECT_EQ(store->moments_bytes_per_block(), expected_per_block);

    const auto moments_path = tmp.path() / "store" / "moments.bin";
    ASSERT_TRUE(fs::exists(moments_path));
    EXPECT_EQ(fs::file_size(moments_path), kNumBlocks * expected_per_block);

    std::vector<std::byte> buf(expected_per_block);
    for (std::size_t b = 0; b < kNumBlocks; ++b) {
        std::fill(buf.begin(), buf.end(), std::byte{0xFF});
        ASSERT_TRUE(store->read_moments(b, std::span<std::byte>(buf)).has_value());
        for (std::size_t i = 0; i < buf.size(); ++i) {
            ASSERT_EQ(buf[i], std::byte{0}) << "block " << b << " byte " << i << " not zero";
        }
    }
}

TEST(BlockStoreMomentsTest, WriteReadRoundTripPerBlock) {
    TempDir tmp;
    constexpr std::size_t kNumBlocks = 4;
    auto store = make_store_with_moments(tmp.path() / "store", kNumBlocks, true);
    ASSERT_NE(store, nullptr);
    const std::size_t per_block = store->moments_bytes_per_block();

    std::vector<std::vector<std::byte>> patterns(kNumBlocks);
    for (std::size_t b = 0; b < kNumBlocks; ++b) {
        patterns[b].assign(per_block, static_cast<std::byte>(0xB0 + b));
        ASSERT_TRUE(store->write_moments(b, std::span<const std::byte>(patterns[b])).has_value());
    }

    std::vector<std::byte> buf(per_block);
    for (std::size_t b = 0; b < kNumBlocks; ++b) {
        ASSERT_TRUE(store->read_moments(b, std::span<std::byte>(buf)).has_value());
        EXPECT_EQ(std::memcmp(buf.data(), patterns[b].data(), per_block), 0)
            << "block " << b << " round-trip mismatch";
    }
}

TEST(BlockStoreMomentsTest, ReopenPreservesMomentsBytes) {
    TempDir tmp;
    constexpr std::size_t kNumBlocks = 2;
    {
        auto store = make_store_with_moments(tmp.path() / "store", kNumBlocks, true);
        ASSERT_NE(store, nullptr);
        std::vector<std::byte> p0(store->moments_bytes_per_block(), std::byte{0xC1});
        std::vector<std::byte> p1(store->moments_bytes_per_block(), std::byte{0xC2});
        ASSERT_TRUE(store->write_moments(0, std::span<const std::byte>(p0)).has_value());
        ASSERT_TRUE(store->write_moments(1, std::span<const std::byte>(p1)).has_value());
    }

    BlockStore::Config cfg;
    cfg.block_size = BlockStore::kDefaultBlockSize;
    auto r2 = BlockStore::open(tmp.path() / "store", cfg);
    ASSERT_TRUE(r2.has_value()) << r2.error();
    auto store2 = std::move(*r2);
    EXPECT_TRUE(store2->has_moments());
    EXPECT_EQ(store2->manifest_version(), 2u);

    std::vector<std::byte> buf(store2->moments_bytes_per_block());
    ASSERT_TRUE(store2->read_moments(0, std::span<std::byte>(buf)).has_value());
    for (auto b : buf) ASSERT_EQ(b, std::byte{0xC1});
    ASSERT_TRUE(store2->read_moments(1, std::span<std::byte>(buf)).has_value());
    for (auto b : buf) ASSERT_EQ(b, std::byte{0xC2});
}

TEST(BlockStoreMomentsTest, OpenLegacyV1ManifestSucceedsWithoutMoments) {
    TempDir tmp;
    const auto dir = tmp.path() / "store";
    fs::create_directories(dir);

    constexpr std::size_t kNumBlocks = 2;
    constexpr std::size_t kBlockSize = BlockStore::kDefaultBlockSize;
    constexpr std::size_t kBytesPerBlock = kBlockSize * BlockStore::kBytesPerGaussian;

    {
        std::vector<std::byte> base(kNumBlocks * kBytesPerBlock, std::byte{0});
        std::ofstream out(dir / "base.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(base.data()),
                  static_cast<std::streamsize>(base.size()));
    }
    {
        std::array<BlockStore::BlockBounds, kNumBlocks> bounds{};
        for (std::size_t b = 0; b < kNumBlocks; ++b) {
            bounds[b].center[0] = static_cast<float>(b);
            bounds[b].radius = 1.0f;
        }
        std::ofstream out(dir / "bounds.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(bounds.data()),
                  static_cast<std::streamsize>(bounds.size() * sizeof(BlockStore::BlockBounds)));
    }
    {
        std::array<BlockStore::IndexEntry, kNumBlocks> entries{};
        for (std::size_t b = 0; b < kNumBlocks; ++b) {
            entries[b].file_id = 0;
            entries[b].offset = static_cast<std::uint64_t>(b) * kBytesPerBlock;
            entries[b].size = static_cast<std::uint32_t>(kBytesPerBlock);
            entries[b].version = 0;
        }
        std::ofstream out(dir / "index.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(entries.data()),
                  static_cast<std::streamsize>(entries.size() * sizeof(BlockStore::IndexEntry)));
    }
    {
        struct LegacyManifest {
            std::uint32_t magic;
            std::uint32_t version;
            std::uint64_t num_blocks;
            std::uint64_t block_size;
            std::uint64_t bytes_per_block;
            std::uint64_t patch_segment_capacity_bytes;
        };
        static_assert(sizeof(LegacyManifest) == 40);
        LegacyManifest m{
            .magic = 0x4C544253u,
            .version = 1,
            .num_blocks = kNumBlocks,
            .block_size = kBlockSize,
            .bytes_per_block = kBytesPerBlock,
            .patch_segment_capacity_bytes = 1ull << 30,
        };
        std::ofstream out(dir / "manifest.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(&m), sizeof(m));
    }

    auto r = BlockStore::open(dir, BlockStore::Config{});
    ASSERT_TRUE(r.has_value()) << r.error();
    auto store = std::move(*r);
    EXPECT_EQ(store->manifest_version(), 1u);
    EXPECT_FALSE(store->has_moments());
    EXPECT_EQ(store->moments_bytes_per_block(), 0u);
    EXPECT_FALSE(fs::exists(dir / "moments.bin"));

    std::vector<std::byte> dummy(BlockStore::kAdamMomentsBytesPerGaussian);
    EXPECT_FALSE(store->read_moments(0, std::span<std::byte>(dummy)).has_value());
    EXPECT_FALSE(store->write_moments(0, std::span<const std::byte>(dummy)).has_value());

    std::vector<std::byte> buf(store->bytes_per_block());
    EXPECT_TRUE(store->read_block(0, std::span<std::byte>(buf)).has_value());
}

TEST(BlockStoreMomentsTest, RejectsBadSizes) {
    TempDir tmp;
    auto store = make_store_with_moments(tmp.path() / "store", 2, true);
    ASSERT_NE(store, nullptr);

    std::vector<std::byte> too_small(store->moments_bytes_per_block() - 1);
    EXPECT_FALSE(store->read_moments(0, std::span<std::byte>(too_small)).has_value());
    EXPECT_FALSE(store->write_moments(0, std::span<const std::byte>(too_small)).has_value());

    std::vector<std::byte> right(store->moments_bytes_per_block());
    EXPECT_FALSE(store->read_moments(99, std::span<std::byte>(right)).has_value());
    EXPECT_FALSE(store->write_moments(99, std::span<const std::byte>(right)).has_value());
}
