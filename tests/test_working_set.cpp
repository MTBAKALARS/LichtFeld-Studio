/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_working_set.cpp
 * @brief Phase 2.5 tests for tide::WorkingSet A/B double-buffer + async prefetch.
 *
 * Requires a live CUDA context. Skips with GTEST_SKIP if cudaSetDevice fails
 * (e.g. CI box with no GPU). Verifies:
 *   - prefetch() + wait_and_activate() round trip across multiple frames
 *   - Retained blocks get D2D-copied (not re-uploaded from host)
 *   - load_and_activate() matches the prefetch + wait pair behaviorally
 *   - Calling prefetch() while a prefetch is already pending is rejected
 *   - active_slices() preserves caller-requested order
 */

#include "core/block_store.hpp"
#include "core/tiered_cache.hpp"
#include "tide/working_set.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <span>
#include <vector>

namespace {

    namespace fs = std::filesystem;
    using lfs::core::BlockStore;
    using lfs::core::TieredCache;
    using lfs::training::tide::WorkingSet;

    // RAII helper: makes a unique temp dir and removes it on scope exit.
    class TempDir {
    public:
        TempDir() {
            std::random_device rd;
            const auto name = "lfs_working_set_test_" + std::to_string(rd()) + "_" +
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

    // Build a synthetic Gaussian cube laid out to produce `num_blocks` blocks
    // after stream_ply_to_base (default block size = 4096). Each Gaussian has
    // its index encoded in `opacity` so per-block bytes are deterministic.
    struct SyntheticData {
        std::vector<float> means, scaling, rotation, opacity, sh0, shN;
        std::size_t n = 0;
        static constexpr std::size_t kRest = 45;
    };

    SyntheticData make_blocks(std::size_t num_blocks) {
        const std::size_t n = BlockStore::kDefaultBlockSize * num_blocks;
        SyntheticData d;
        d.n = n;
        d.means.assign(3 * n, 0.0f);
        d.scaling.assign(3 * n, 0.0f);
        d.rotation.assign(4 * n, 0.0f);
        d.opacity.assign(n, 0.0f);
        d.sh0.assign(3 * n, 0.0f);
        d.shN.assign(SyntheticData::kRest * n, 0.0f);

        // Lay positions on a deterministic 1D ramp so Morton ordering is stable.
        for (std::size_t i = 0; i < n; ++i) {
            d.means[3 * i + 0] = static_cast<float>(i % 64);
            d.means[3 * i + 1] = static_cast<float>((i / 64) % 64);
            d.means[3 * i + 2] = static_cast<float>(i / (64 * 64));
            d.rotation[4 * i + 0] = 1.0f;
            d.opacity[i] = static_cast<float>(i);
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

    static std::shared_ptr<BlockStore> share(std::unique_ptr<BlockStore> p) {
        return std::shared_ptr<BlockStore>(std::move(p));
    }

    // Returns true iff a CUDA device is usable.
    bool cuda_available() {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess) return false;
        if (count <= 0) return false;
        return cudaSetDevice(0) == cudaSuccess;
    }

    // RAII fixture: temp dir + store + cache + working set sized for `capacity`
    // resident blocks. Tests typically create with 4-8 source blocks and a
    // smaller capacity to exercise eviction.
    struct WSFixture {
        TempDir tmp;
        std::shared_ptr<BlockStore> store;
        std::unique_ptr<TieredCache> cache;
        std::unique_ptr<WorkingSet> ws;

        WSFixture(std::size_t num_blocks, std::size_t capacity) {
            auto data = make_blocks(num_blocks);
            auto r = BlockStore::stream_ply_to_base(tmp.path() / "store", as_source(data));
            if (!r.has_value()) {
                ADD_FAILURE() << "stream_ply_to_base: " << r.error();
                return;
            }
            store = share(std::move(r.value()));

            TieredCache::Config ccfg;
            ccfg.capacity_blocks = std::max<std::size_t>(capacity * 2, 4);
            cache = std::make_unique<TieredCache>(store, ccfg);

            WorkingSet::Config wcfg;
            wcfg.capacity_blocks = capacity;
            wcfg.bytes_per_block = store->bytes_per_block();
            wcfg.cuda_device = 0;
            auto wr = WorkingSet::create(wcfg);
            if (!wr.has_value()) {
                ADD_FAILURE() << "WorkingSet::create: " << wr.error();
                return;
            }
            ws = std::move(wr.value());
        }
    };

} // namespace

// ============================================================================
// Tests
// ============================================================================

TEST(WorkingSetTest, PrefetchActivateRoundTrip) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA device not available";
    WSFixture f(/*num_blocks=*/8, /*capacity=*/4);
    ASSERT_TRUE(f.ws);

    // Frame 0: blocks 0..3.
    const std::array<std::size_t, 4> frame0 = {0, 1, 2, 3};
    auto p0 = f.ws->prefetch(*f.cache, std::span<const std::size_t>(frame0));
    ASSERT_TRUE(p0.has_value()) << p0.error();
    EXPECT_TRUE(f.ws->prefetch_pending());

    auto a0 = f.ws->wait_and_activate();
    ASSERT_TRUE(a0.has_value()) << a0.error();
    EXPECT_FALSE(f.ws->prefetch_pending());
    EXPECT_EQ(f.ws->active_block_count(), 4u);
    EXPECT_NE(f.ws->device_buffer(), nullptr);

    // Active slice order matches request.
    auto slices = f.ws->active_slices();
    ASSERT_EQ(slices.size(), 4u);
    for (std::size_t i = 0; i < 4; ++i) EXPECT_EQ(slices[i].block_id, frame0[i]);

    // Frame 0 had 4 fresh uploads, 0 retains.
    const auto s0 = f.ws->stats();
    EXPECT_EQ(s0.prefetches, 1u);
    EXPECT_EQ(s0.activates, 1u);
    EXPECT_EQ(s0.blocks_uploaded, 4u);
    EXPECT_EQ(s0.blocks_retained, 0u);
}

TEST(WorkingSetTest, RetainedBlocksUseD2DCopy) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA device not available";
    WSFixture f(/*num_blocks=*/8, /*capacity=*/4);
    ASSERT_TRUE(f.ws);

    const std::array<std::size_t, 4> frame0 = {0, 1, 2, 3};
    ASSERT_TRUE(f.ws->load_and_activate(*f.cache, frame0).has_value());

    const auto s_after_f0 = f.ws->stats();
    ASSERT_EQ(s_after_f0.blocks_uploaded, 4u);
    ASSERT_EQ(s_after_f0.blocks_retained, 0u);

    // Frame 1: blocks {2, 3, 4, 5}. Overlap = {2, 3} (2 retained, 2 new).
    const std::array<std::size_t, 4> frame1 = {2, 3, 4, 5};
    ASSERT_TRUE(f.ws->load_and_activate(*f.cache, frame1).has_value());

    const auto s_after_f1 = f.ws->stats();
    EXPECT_EQ(s_after_f1.blocks_uploaded, s_after_f0.blocks_uploaded + 2u);
    EXPECT_EQ(s_after_f1.blocks_retained, s_after_f0.blocks_retained + 2u);
    EXPECT_GT(s_after_f1.bytes_d2d_copied, 0u);

    // Verify active set matches the new request order.
    auto slices = f.ws->active_slices();
    ASSERT_EQ(slices.size(), 4u);
    for (std::size_t i = 0; i < 4; ++i) EXPECT_EQ(slices[i].block_id, frame1[i]);

    // Frame 2: blocks {0, 1, 2, 3} — fully disjoint from frame1 except {2, 3}.
    // Should retain 2 blocks again (the {2,3} overlap with frame1).
    const std::array<std::size_t, 4> frame2 = {0, 1, 2, 3};
    ASSERT_TRUE(f.ws->load_and_activate(*f.cache, frame2).has_value());
    const auto s_after_f2 = f.ws->stats();
    EXPECT_EQ(s_after_f2.blocks_retained, s_after_f1.blocks_retained + 2u);
    EXPECT_EQ(s_after_f2.blocks_uploaded, s_after_f1.blocks_uploaded + 2u);
}

TEST(WorkingSetTest, SecondPrefetchWithoutActivateIsRejected) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA device not available";
    WSFixture f(/*num_blocks=*/4, /*capacity=*/4);
    ASSERT_TRUE(f.ws);

    const std::array<std::size_t, 2> ids_a = {0, 1};
    ASSERT_TRUE(f.ws->prefetch(*f.cache, ids_a).has_value());
    ASSERT_TRUE(f.ws->prefetch_pending());

    const std::array<std::size_t, 2> ids_b = {2, 3};
    auto p2 = f.ws->prefetch(*f.cache, ids_b);
    EXPECT_FALSE(p2.has_value());
    EXPECT_NE(p2.error().find("not yet activated"), std::string::npos)
        << "error message was: " << p2.error();

    // Recover by activating, then a fresh prefetch must succeed.
    ASSERT_TRUE(f.ws->wait_and_activate().has_value());
    EXPECT_FALSE(f.ws->prefetch_pending());
    ASSERT_TRUE(f.ws->prefetch(*f.cache, ids_b).has_value());
    ASSERT_TRUE(f.ws->wait_and_activate().has_value());

    auto slices = f.ws->active_slices();
    ASSERT_EQ(slices.size(), 2u);
    EXPECT_EQ(slices[0].block_id, 2u);
    EXPECT_EQ(slices[1].block_id, 3u);
}

TEST(WorkingSetTest, ActivateWithoutPrefetchFails) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA device not available";
    WSFixture f(/*num_blocks=*/4, /*capacity=*/4);
    ASSERT_TRUE(f.ws);

    auto r = f.ws->wait_and_activate();
    EXPECT_FALSE(r.has_value());
    EXPECT_NE(r.error().find("no prefetch"), std::string::npos)
        << "error message was: " << r.error();
}

TEST(WorkingSetTest, ExceedingCapacityIsRejected) {
    if (!cuda_available()) GTEST_SKIP() << "CUDA device not available";
    WSFixture f(/*num_blocks=*/8, /*capacity=*/4);
    ASSERT_TRUE(f.ws);

    const std::array<std::size_t, 5> too_many = {0, 1, 2, 3, 4};
    auto r = f.ws->prefetch(*f.cache, too_many);
    EXPECT_FALSE(r.has_value());
    EXPECT_NE(r.error().find("capacity"), std::string::npos)
        << "error message was: " << r.error();
}
