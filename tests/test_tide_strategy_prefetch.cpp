/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_tide_strategy_prefetch.cpp
 * @brief Phase 3.5.6a: TideStrategy pipelined-prefetch state machine.
 *
 * Exercises the strategy-level `prefetch_next` hook end-to-end through the
 * matching `pre_forward` consumption path:
 *
 *   - prefetch_next stages a predicted resident set onto the inactive
 *     WorkingSet buffer (issued async, observable via `has_pending_prefetch`).
 *   - pre_forward with matching ids consumes it via wait_and_activate
 *     (counter: prefetch_hits).
 *   - pre_forward with mismatched ids drains the stale prefetch then
 *     sync-loads the correct set (counters: prefetch_misses + sync_loads).
 *   - prefetch_next is a no-op when the predicted set equals last_loaded_ids
 *     (counter: prefetch_skipped_no_change).
 *   - A second prefetch_next while one is in flight is a no-op (the
 *     WorkingSet contract forbids issuing two outstanding prefetches).
 *
 * All scenarios run on a v2 store with moments enabled and a capacity
 * strictly less than the block count so Mode C engages — that is the only
 * residency policy that can produce *different* resident sets across calls.
 *
 * Skips with GTEST_SKIP if no CUDA device is present.
 */

#include "core/block_store.hpp"
#include "core/camera.hpp"
#include "core/camera_types.h"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/tiered_cache.hpp"
#include "strategies/tide_strategy.hpp"
#include "tide/tide_runtime.hpp"
#include "tide/working_set.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <span>
#include <unordered_set>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    namespace fs = std::filesystem;

    bool cuda_available() {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess) return false;
        if (count <= 0) return false;
        return cudaSetDevice(0) == cudaSuccess;
    }

    class TempDir {
    public:
        TempDir() {
            std::random_device rd;
            const auto name = "lfs_tide_strategy_prefetch_" + std::to_string(rd()) + "_" + std::to_string(rd());
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

    constexpr std::size_t kRest = 45; // SH-3
    constexpr int kShDegree = 3;

    // Same synthetic-store layout as test_tide_strategy_lru.cpp. Blocks are
    // spread along +X so the frustum culler sees a contiguous subset; we
    // override per-block bounds via `force_visibility` for determinism.
    void make_synthetic_store(const fs::path& dir, std::size_t num_blocks, bool with_moments) {
        const std::size_t n = BlockStore::kDefaultBlockSize * num_blocks;
        std::vector<float> means(3 * n, 0.0f);
        std::vector<float> scaling(3 * n, -2.0f);
        std::vector<float> rotation(4 * n, 0.0f);
        std::vector<float> opacity(n, 0.5f);
        std::vector<float> sh0(3 * n, 0.5f);
        std::vector<float> shN(kRest * n, 0.0f);
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t block_index = i / BlockStore::kDefaultBlockSize;
            const std::size_t intra = i % BlockStore::kDefaultBlockSize;
            means[3 * i + 0] = static_cast<float>(block_index) * 1000.0f
                             + static_cast<float>(intra % 16) * 0.1f;
            means[3 * i + 1] = static_cast<float>((intra / 16) % 16) * 0.1f;
            means[3 * i + 2] = static_cast<float>(intra / (16 * 16)) * 0.1f;
            rotation[4 * i + 0] = 1.0f;
        }
        BlockStore::PlySource src;
        src.means = means;
        src.scaling = scaling;
        src.rotation = rotation;
        src.opacity = opacity;
        src.sh0 = sh0;
        src.shN = shN;
        src.num_gaussians = n;
        src.sh_rest_components = kRest;

        BlockStore::Config cfg;
        cfg.with_moments = with_moments;
        auto r = BlockStore::stream_ply_to_base(dir, src, cfg);
        ASSERT_TRUE(r.has_value()) << "stream_ply_to_base: " << r.error();
    }

    SplatData make_placeholder() {
        constexpr std::size_t kN = 4;
        std::vector<float> means(kN * 3, 0.0f);
        std::vector<float> sh0(kN * 3, 0.5f);
        std::vector<float> shN(kN * kRest, 0.0f);
        std::vector<float> scaling(kN * 3, -2.0f);
        std::vector<float> rotation(kN * 4, 0.0f);
        std::vector<float> opacity(kN, 0.5f);
        for (std::size_t i = 0; i < kN; ++i) rotation[4 * i] = 1.0f;
        return SplatData(kShDegree,
                         Tensor::from_vector(means, TensorShape({kN, 3}), Device::CUDA),
                         Tensor::from_vector(sh0, TensorShape({kN, 1, 3}), Device::CUDA),
                         Tensor::from_vector(shN, TensorShape({kN, kRest / 3, 3}), Device::CUDA),
                         Tensor::from_vector(scaling, TensorShape({kN, 3}), Device::CUDA),
                         Tensor::from_vector(rotation, TensorShape({kN, 4}), Device::CUDA),
                         Tensor::from_vector(opacity, TensorShape({kN, 1}), Device::CUDA),
                         1.0f);
    }

    Camera make_camera_at(float tx) {
        const std::vector<float> R_data = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        const std::vector<float> T_data = {-tx, 0, 4};
        auto R = Tensor::from_blob(const_cast<float*>(R_data.data()),
                                   {3, 3}, Device::CPU, DataType::Float32)
                     .to(Device::CUDA);
        auto T = Tensor::from_blob(const_cast<float*>(T_data.data()),
                                   {3}, Device::CPU, DataType::Float32)
                     .to(Device::CUDA);
        return Camera(R, T,
                      /*fx=*/600.0f, /*fy=*/600.0f,
                      /*cx=*/320.0f, /*cy=*/240.0f,
                      Tensor(), Tensor(),
                      CameraModelType::PINHOLE,
                      /*image_name=*/"prefetch_cam",
                      /*image_path=*/"",
                      /*mask_path=*/std::filesystem::path{},
                      /*camera_width=*/640, /*camera_height=*/480,
                      /*uid=*/0);
    }

    param::OptimizationParameters make_opt_params(const fs::path& store_dir,
                                                  std::size_t capacity) {
        param::OptimizationParameters p;
        p.iterations = 16;
        p.max_cap = 256;
        p.tide_store_path = store_dir;
        p.tide_capacity_blocks = capacity;
        return p;
    }

    // Overwrite per-block bounds so FrustumCuller returns exactly `visible_ids`
    // for a camera at world (0,0,-4) looking +Z. Visible blocks get a tiny
    // sphere in front of the camera; non-visible blocks are exiled to +1e6 m
    // along +X (far outside any reasonable frustum). Same construction as
    // test_tide_strategy_lru.cpp::force_visibility.
    void force_visibility(BlockStore& store,
                          std::size_t num_blocks,
                          const std::vector<std::size_t>& visible_ids) {
        std::unordered_set<std::size_t> vis_set(visible_ids.begin(), visible_ids.end());
        for (std::size_t bid = 0; bid < num_blocks; ++bid) {
            BlockStore::BlockBounds b{};
            if (vis_set.count(bid)) {
                b.center[0] = 0.0f;
                b.center[1] = 0.0f;
                b.center[2] = 5.0f;
                b.radius = 0.1f;
            } else {
                b.center[0] = 1.0e6f;
                b.center[1] = 0.0f;
                b.center[2] = 0.0f;
                b.radius = 0.1f;
            }
            store.update_bounds(bid, b);
        }
    }

} // namespace

#define SKIP_IF_NO_CUDA()                              \
    do {                                               \
        if (!cuda_available()) {                       \
            GTEST_SKIP() << "No CUDA device present."; \
        }                                              \
    } while (0)

// ============================================================================
// `prefetch_next` stages the predicted resident set onto the inactive WS
// buffer and bumps `prefetches_issued`. No `pre_forward` precedes it so
// `last_loaded_ids` is empty — the equality-skip short-circuit must NOT fire.
// ============================================================================

TEST(TideStrategyPrefetchTest, PrefetchNextStagesNextSet) {
    SKIP_IF_NO_CUDA();
    TempDir tmp;
    const fs::path store_dir = tmp.path() / "store";
    constexpr std::size_t kNumBlocks = 6;
    constexpr std::size_t kCapacity = 4;
    make_synthetic_store(store_dir, kNumBlocks, /*with_moments=*/true);

    auto placeholder = make_placeholder();
    TideStrategy strategy(placeholder);
    auto opt = make_opt_params(store_dir, kCapacity);
    auto rt = lfs::training::tide::attach_tide_working_set(strategy, opt);
    ASSERT_TRUE(rt.has_value()) << rt.error();
    auto runtime = std::move(rt.value());
    strategy.initialize(opt);

    force_visibility(*runtime.store, kNumBlocks, /*visible_ids=*/{0, 1});

    EXPECT_FALSE(strategy.has_pending_prefetch())
        << "No prefetch should be pending before any prefetch_next call.";

    auto cam = make_camera_at(0.0f);
    ASSERT_NO_THROW(strategy.prefetch_next(0, cam));

    EXPECT_TRUE(strategy.has_pending_prefetch())
        << "prefetch_next must leave a pending WorkingSet prefetch.";

    const auto stats = strategy.prefetch_stats();
    EXPECT_EQ(stats.prefetches_issued, 1u);
    EXPECT_EQ(stats.prefetch_hits, 0u);
    EXPECT_EQ(stats.prefetch_misses, 0u);
    EXPECT_EQ(stats.sync_loads, 0u);
    EXPECT_EQ(stats.prefetch_skipped_no_change, 0u);

    const auto& staged = strategy.last_prefetched_ids();
    ASSERT_EQ(staged.size(), kCapacity)
        << "Mode C engaged with capacity=" << kCapacity
        << " so the predicted resident set must fill capacity exactly.";

    // The staged set must contain every visible block.
    const std::unordered_set<std::size_t> staged_set(staged.begin(), staged.end());
    EXPECT_TRUE(staged_set.count(0)) << "Visible block 0 not staged.";
    EXPECT_TRUE(staged_set.count(1)) << "Visible block 1 not staged.";
}

// ============================================================================
// Happy path: `prefetch_next` followed by a matching `pre_forward` consumes
// the staged buffer via wait_and_activate (prefetch_hits++), and the slow
// `sync_loads` path does NOT fire.
// ============================================================================

TEST(TideStrategyPrefetchTest, PreForwardWaitsOnMatchingPrefetch) {
    SKIP_IF_NO_CUDA();
    TempDir tmp;
    const fs::path store_dir = tmp.path() / "store";
    constexpr std::size_t kNumBlocks = 6;
    constexpr std::size_t kCapacity = 4;
    make_synthetic_store(store_dir, kNumBlocks, /*with_moments=*/true);

    auto placeholder = make_placeholder();
    TideStrategy strategy(placeholder);
    auto opt = make_opt_params(store_dir, kCapacity);
    auto rt = lfs::training::tide::attach_tide_working_set(strategy, opt);
    ASSERT_TRUE(rt.has_value()) << rt.error();
    auto runtime = std::move(rt.value());
    strategy.initialize(opt);

    force_visibility(*runtime.store, kNumBlocks, /*visible_ids=*/{0, 1});

    auto cam = make_camera_at(0.0f);
    ASSERT_NO_THROW(strategy.prefetch_next(0, cam));
    ASSERT_TRUE(strategy.has_pending_prefetch());

    // pre_forward with the same (iter, camera) MUST consume the prefetch
    // — visibility hasn't changed, so the predicted set matches the actual
    // resident_scratch and we hit the wait_and_activate "match" branch.
    ASSERT_NO_THROW(strategy.pre_forward(0, cam));

    EXPECT_FALSE(strategy.has_pending_prefetch())
        << "pre_forward must consume the pending prefetch.";
    EXPECT_TRUE(strategy.last_pre_forward_loaded())
        << "Consuming a prefetch counts as a load (wait_and_activate swapped buffers).";

    const auto stats = strategy.prefetch_stats();
    EXPECT_EQ(stats.prefetch_hits, 1u) << "Matching prefetch must bump prefetch_hits.";
    EXPECT_EQ(stats.prefetch_misses, 0u);
    EXPECT_EQ(stats.sync_loads, 0u)
        << "Sync load_and_activate path must NOT fire when prefetch ids match.";

    // last_prefetched_ids cleared once consumed.
    EXPECT_TRUE(strategy.last_prefetched_ids().empty())
        << "pending_prefetch_ids must be cleared after wait_and_activate.";
}

// ============================================================================
// Sad path: prefetch_next stages set A, but visibility changes before
// pre_forward computes set B (≠ A). pre_forward must (a) drain the stale
// prefetch via wait_and_activate, (b) bump prefetch_misses, (c) fall through
// to sync `load_and_activate` of the correct set (sync_loads++).
// ============================================================================

TEST(TideStrategyPrefetchTest, PreForwardFallsBackToSyncOnMismatch) {
    SKIP_IF_NO_CUDA();
    TempDir tmp;
    const fs::path store_dir = tmp.path() / "store";
    constexpr std::size_t kNumBlocks = 6;
    constexpr std::size_t kCapacity = 4;
    make_synthetic_store(store_dir, kNumBlocks, /*with_moments=*/true);

    auto placeholder = make_placeholder();
    TideStrategy strategy(placeholder);
    auto opt = make_opt_params(store_dir, kCapacity);
    auto rt = lfs::training::tide::attach_tide_working_set(strategy, opt);
    ASSERT_TRUE(rt.has_value()) << rt.error();
    auto runtime = std::move(rt.value());
    strategy.initialize(opt);

    // Stage A predicted from visibility {0, 1}.
    force_visibility(*runtime.store, kNumBlocks, /*visible_ids=*/{0, 1});
    auto cam = make_camera_at(0.0f);
    ASSERT_NO_THROW(strategy.prefetch_next(0, cam));
    ASSERT_TRUE(strategy.has_pending_prefetch());
    const std::vector<std::size_t> staged_A = strategy.last_prefetched_ids();

    // Change visibility to {3, 4} before pre_forward computes set B.
    // Since the LRU fillers depend on `block_last_used_iter` (all -1 here),
    // the new resident set will be {3, 4} + 2 of the never-resident blocks
    // — distinct from staged_A which had {0, 1} as the visible anchors.
    force_visibility(*runtime.store, kNumBlocks, /*visible_ids=*/{3, 4});
    ASSERT_NO_THROW(strategy.pre_forward(0, cam));

    EXPECT_FALSE(strategy.has_pending_prefetch())
        << "pre_forward must drain the stale prefetch.";
    EXPECT_TRUE(strategy.last_pre_forward_loaded());

    const auto stats = strategy.prefetch_stats();
    EXPECT_EQ(stats.prefetch_misses, 1u) << "Mismatched prefetch must bump prefetch_misses.";
    EXPECT_EQ(stats.prefetch_hits, 0u);
    EXPECT_EQ(stats.sync_loads, 1u) << "Mismatch must fall through to a sync load_and_activate.";

    const auto& resident = strategy.last_resident_block_ids();
    const std::unordered_set<std::size_t> resident_set(resident.begin(), resident.end());
    EXPECT_TRUE(resident_set.count(3));
    EXPECT_TRUE(resident_set.count(4));
    EXPECT_NE(resident, staged_A)
        << "Mismatch test is meaningless if the stale prefetch happened to match.";
}

// ============================================================================
// Equality-skip on the producer side: when `predicted == last_loaded_ids`,
// `prefetch_next` must NOT issue a WorkingSet prefetch (no wasted I/O) —
// `pre_forward` would short-circuit the load anyway. The skip counter
// (`prefetch_skipped_no_change`) tracks this.
// ============================================================================

TEST(TideStrategyPrefetchTest, PrefetchNextSkippedWhenPredictedEqualsLoaded) {
    SKIP_IF_NO_CUDA();
    TempDir tmp;
    const fs::path store_dir = tmp.path() / "store";
    constexpr std::size_t kNumBlocks = 6;
    constexpr std::size_t kCapacity = 4;
    make_synthetic_store(store_dir, kNumBlocks, /*with_moments=*/true);

    auto placeholder = make_placeholder();
    TideStrategy strategy(placeholder);
    auto opt = make_opt_params(store_dir, kCapacity);
    auto rt = lfs::training::tide::attach_tide_working_set(strategy, opt);
    ASSERT_TRUE(rt.has_value()) << rt.error();
    auto runtime = std::move(rt.value());
    strategy.initialize(opt);

    force_visibility(*runtime.store, kNumBlocks, /*visible_ids=*/{0, 1});
    auto cam = make_camera_at(0.0f);

    // Cold sync load establishes last_loaded_ids.
    ASSERT_NO_THROW(strategy.pre_forward(0, cam));
    EXPECT_TRUE(strategy.last_pre_forward_loaded());
    {
        const auto s = strategy.prefetch_stats();
        EXPECT_EQ(s.sync_loads, 1u);
        EXPECT_EQ(s.prefetches_issued, 0u);
    }

    // prefetch_next with the same camera — predicted == last_loaded → skip.
    ASSERT_NO_THROW(strategy.prefetch_next(1, cam));
    EXPECT_FALSE(strategy.has_pending_prefetch())
        << "prefetch_next must short-circuit when predicted matches last_loaded.";

    const auto stats = strategy.prefetch_stats();
    EXPECT_EQ(stats.prefetch_skipped_no_change, 1u);
    EXPECT_EQ(stats.prefetches_issued, 0u)
        << "No WorkingSet::prefetch should have been issued.";
    EXPECT_EQ(stats.prefetch_hits, 0u);
    EXPECT_EQ(stats.prefetch_misses, 0u);
}

// ============================================================================
// Double-prefetch guard: the WorkingSet contract forbids a second prefetch
// before the first completes. `prefetch_next` must silently no-op while a
// previous prefetch is pending — the first staged set stays put.
// ============================================================================

TEST(TideStrategyPrefetchTest, PrefetchNextNoopWhileAnotherIsPending) {
    SKIP_IF_NO_CUDA();
    TempDir tmp;
    const fs::path store_dir = tmp.path() / "store";
    constexpr std::size_t kNumBlocks = 6;
    constexpr std::size_t kCapacity = 4;
    make_synthetic_store(store_dir, kNumBlocks, /*with_moments=*/true);

    auto placeholder = make_placeholder();
    TideStrategy strategy(placeholder);
    auto opt = make_opt_params(store_dir, kCapacity);
    auto rt = lfs::training::tide::attach_tide_working_set(strategy, opt);
    ASSERT_TRUE(rt.has_value()) << rt.error();
    auto runtime = std::move(rt.value());
    strategy.initialize(opt);

    auto cam = make_camera_at(0.0f);

    // First prefetch: vis={0, 1}.
    force_visibility(*runtime.store, kNumBlocks, /*visible_ids=*/{0, 1});
    ASSERT_NO_THROW(strategy.prefetch_next(0, cam));
    ASSERT_TRUE(strategy.has_pending_prefetch());
    const std::vector<std::size_t> first_staged = strategy.last_prefetched_ids();
    ASSERT_FALSE(first_staged.empty());

    // Second prefetch with DIFFERENT visibility while first still pending.
    // Must be a no-op: prefetches_issued stays at 1, last_prefetched_ids
    // unchanged (still reflects the in-flight set).
    force_visibility(*runtime.store, kNumBlocks, /*visible_ids=*/{3, 4});
    ASSERT_NO_THROW(strategy.prefetch_next(1, cam));

    EXPECT_TRUE(strategy.has_pending_prefetch())
        << "The original prefetch must still be pending.";
    const auto stats = strategy.prefetch_stats();
    EXPECT_EQ(stats.prefetches_issued, 1u)
        << "Double prefetch_next must not issue a second WS::prefetch.";

    const auto& staged_now = strategy.last_prefetched_ids();
    EXPECT_EQ(staged_now, first_staged)
        << "pending_prefetch_ids must reflect the original (in-flight) set, "
           "not the would-be second one.";
}
