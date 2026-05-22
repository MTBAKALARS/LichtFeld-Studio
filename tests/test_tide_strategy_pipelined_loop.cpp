/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_tide_strategy_pipelined_loop.cpp
 * @brief Phase 3.5.6b: multi-iteration trainer pipelined-loop pattern.
 *
 * Simulates the exact peek+stage+consume sequence the Trainer's per-iteration
 * loop runs in Phase 3.5.6b:
 *
 *   iter N start:    consume stashed example for iter N (or sync-load on cold).
 *   iter N body:     pre_forward(N, *cam_N)        -- forward+backward+step.
 *   iter N tail:     example_{N+1} = dataloader->next();
 *                    strategy_->prefetch_next(N+1, *cam_{N+1});
 *   iter N+1 start:  pre_forward(N+1, *cam_{N+1})  -- consumes staged set.
 *
 * The strategy-level state machine is exhaustively covered by
 * test_tide_strategy_prefetch.cpp. The job of this file is to confirm that
 * the *loop pattern itself* — repeatedly calling prefetch_next(iter+1, next_cam)
 * at end-of-iter and then pre_forward(iter+1, next_cam) at the start of the
 * next iter — produces the expected steady-state counter pattern:
 *
 *     iter 0 : cold sync_load        (sync_loads = 1)
 *     iter k>=1: prefetch_hit         (prefetch_hits += 1 each iter)
 *
 * With three iterations and three distinct visible sets the steady state is
 * sync_loads=1, prefetch_hits=2, prefetches_issued=2, prefetch_misses=0.
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
            const auto name = "lfs_tide_strategy_pipelined_loop_" +
                              std::to_string(rd()) + "_" + std::to_string(rd());
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

    Camera make_camera(int uid) {
        const std::vector<float> R_data = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        const std::vector<float> T_data = {0, 0, 4};
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
                      /*image_name=*/"pipeline_cam_" + std::to_string(uid),
                      /*image_path=*/"",
                      /*mask_path=*/std::filesystem::path{},
                      /*camera_width=*/640, /*camera_height=*/480,
                      /*uid=*/uid);
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
// Three-iteration trainer-style pipelined loop with the visible set rotating
// each iter. We pick CAPACITY-EXHAUSTING visible sets so the resident set is
// exactly {visible} (no LRU-fill tie-breaking can move it around):
//
//   iter 0 visible: {0,1,2,3}  -> resident {0,1,2,3}
//   iter 1 visible: {2,3,4,5}  -> resident {2,3,4,5}
//   iter 2 visible: {0,1,4,5}  -> resident {0,1,4,5}
//
// Capacity == |visible| == 4 means `remaining == 0` in compute_resident_set_
// and the LRU-fill branch is skipped entirely. Predicted resident sets at
// the prefetch_next site match exactly what pre_forward computes for the
// next iter, and they always differ from the previous iter's last_loaded
// (so the equality-skip short-circuit never fires).
//
// The full state machine (mismatch -> sync_load, double-prefetch no-op,
// equality skip) is covered in test_tide_strategy_prefetch.cpp.
// ============================================================================

TEST(TideStrategyPipelinedLoopTest, ThreeIterPipelinedLoopHitsEveryIterAfterColdStart) {
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

    Camera cam0 = make_camera(0);
    Camera cam1 = make_camera(1);
    Camera cam2 = make_camera(2);

    // ----- iter 0 (cold start) -----
    // Trainer has no stashed example; it pulls from dataloader and calls
    // pre_forward synchronously. We simulate that here.
    force_visibility(*runtime.store, kNumBlocks, /*visible_ids=*/{0, 1, 2, 3});
    ASSERT_NO_THROW(strategy.pre_forward(0, cam0));
    EXPECT_TRUE(strategy.last_pre_forward_loaded())
        << "Cold start at iter 0 must perform a sync load_and_activate.";

    {
        const auto s = strategy.prefetch_stats();
        EXPECT_EQ(s.sync_loads, 1u);
        EXPECT_EQ(s.prefetches_issued, 0u);
        EXPECT_EQ(s.prefetch_hits, 0u);
        EXPECT_EQ(s.prefetch_misses, 0u);
    }

    // ----- iter 0 tail: peek + prefetch for iter 1 -----
    // Visibility for the NEXT iteration must be set before the peek so that
    // prefetch_next computes the same predicted set that pre_forward will
    // compute at the top of iter 1.
    force_visibility(*runtime.store, kNumBlocks, /*visible_ids=*/{2, 3, 4, 5});
    ASSERT_NO_THROW(strategy.prefetch_next(1, cam1));
    EXPECT_TRUE(strategy.has_pending_prefetch())
        << "End-of-iter prefetch_next must stage onto the inactive WS buffer.";

    // ----- iter 1 start: consume stashed example + pre_forward -----
    ASSERT_NO_THROW(strategy.pre_forward(1, cam1));
    EXPECT_TRUE(strategy.last_pre_forward_loaded())
        << "wait_and_activate is a load (counters as 'loaded' in the strategy).";
    EXPECT_FALSE(strategy.has_pending_prefetch());

    {
        const auto s = strategy.prefetch_stats();
        EXPECT_EQ(s.sync_loads, 1u) << "Only iter 0 should sync-load.";
        EXPECT_EQ(s.prefetches_issued, 1u);
        EXPECT_EQ(s.prefetch_hits, 1u)
            << "Iter 1 pre_forward must consume the prefetch staged at iter 0 tail.";
        EXPECT_EQ(s.prefetch_misses, 0u);
    }

    // ----- iter 1 tail: peek + prefetch for iter 2 -----
    force_visibility(*runtime.store, kNumBlocks, /*visible_ids=*/{0, 1, 4, 5});
    ASSERT_NO_THROW(strategy.prefetch_next(2, cam2));
    EXPECT_TRUE(strategy.has_pending_prefetch());

    // ----- iter 2 start: consume + pre_forward -----
    ASSERT_NO_THROW(strategy.pre_forward(2, cam2));
    EXPECT_TRUE(strategy.last_pre_forward_loaded());
    EXPECT_FALSE(strategy.has_pending_prefetch());

    // Steady-state assertion: every iter after the cold start hit the
    // prefetch path, and no sync re-loads or misses occurred.
    const auto s = strategy.prefetch_stats();
    EXPECT_EQ(s.sync_loads, 1u)
        << "Steady-state pipelined loop must keep sync_loads pinned at the "
           "single cold-start load.";
    EXPECT_EQ(s.prefetches_issued, 2u);
    EXPECT_EQ(s.prefetch_hits, 2u)
        << "Both end-of-iter prefetches must be hit by the next iter's pre_forward.";
    EXPECT_EQ(s.prefetch_misses, 0u)
        << "No mismatches: visibility is set before the peek so predicted == actual.";
}

// ============================================================================
// Trainer's "stop-at-end-of-iter" path: when the loop is about to break out
// (e.g. stop_token requested between iter N's train_step and the peek), the
// peek+prefetch_next must NOT have been called, the strategy must hold no
// pending prefetch, and no counters except those from the in-flight iter must
// have moved.
//
// This is a *negative* test that mirrors the trainer-side `if (... stop ...)
// continue around the prefetch peek` guard introduced in 3.5.6b.
// ============================================================================

TEST(TideStrategyPipelinedLoopTest, NoPrefetchIfTrainerSkipsPeek) {
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

    Camera cam0 = make_camera(0);

    force_visibility(*runtime.store, kNumBlocks, /*visible_ids=*/{0, 1});
    ASSERT_NO_THROW(strategy.pre_forward(0, cam0));

    // Trainer hits Stop / stop_token after train_step -> does NOT call
    // prefetch_next. State machine must remain quiescent.
    EXPECT_FALSE(strategy.has_pending_prefetch());

    const auto s = strategy.prefetch_stats();
    EXPECT_EQ(s.prefetches_issued, 0u);
    EXPECT_EQ(s.prefetch_hits, 0u);
    EXPECT_EQ(s.prefetch_misses, 0u);
    EXPECT_EQ(s.sync_loads, 1u) << "Cold-start sync load only.";
    EXPECT_EQ(s.prefetch_skipped_no_change, 0u);
}
