/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_tide_strategy_lru.cpp
 * @brief Phase 3.5.4: TideStrategy LRU residency policy with visibility protection.
 *
 * Three modes verified end-to-end through pre_forward:
 *   - **Mode A (legacy v1)**: no moments sidecar -> iota-all-blocks fast path,
 *     `last_pre_forward_used_lru()` reports false. Variable residency is
 *     deliberately disabled because TideResidentAdam's internal m/v buffers
 *     are flat-indexed by SOA position.
 *   - **Mode B (v2 fits)**: moments enabled AND num_blocks <= capacity -> still
 *     iota-all. No eviction needed.
 *   - **Mode C (v2 over capacity)**: moments enabled AND num_blocks > capacity ->
 *     visible-first + LRU fill. Visible blocks always resident; remaining
 *     capacity filled with non-visible blocks by descending `last_used_iter`.
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
            const auto name = "lfs_tide_strategy_lru_" + std::to_string(rd()) + "_" + std::to_string(rd());
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

    // Bake a synthetic store with `num_blocks` blocks, optionally with the
    // moments sidecar enabled. Blocks are spread along +X (each block 1000 m
    // apart at intra-block scale 0.1 m) so a frustum culler sees a contiguous
    // subset depending on camera placement.
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
                      /*image_name=*/"lru_cam",
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

    // Overwrite the BlockStore's per-block bounding spheres so the frustum
    // culler used inside `pre_forward` returns a deterministic visible set.
    // Blocks in `visible_ids` get tiny spheres placed in front of the test
    // camera (which sits at world (0,0,-4) looking +Z); all other blocks are
    // moved to (+1e6, 0, 0) which is far outside the right/top/left/bottom
    // planes regardless of FoV. This insulates the LRU policy assertions from
    // any subtle behavior of `FrustumCuller` on the synthetic store geometry.
    void force_visibility(BlockStore& store,
                          std::size_t num_blocks,
                          const std::vector<std::size_t>& visible_ids) {
        std::unordered_set<std::size_t> vis_set(visible_ids.begin(), visible_ids.end());
        for (std::size_t bid = 0; bid < num_blocks; ++bid) {
            BlockStore::BlockBounds b{};
            if (vis_set.count(bid)) {
                // Visible: small sphere ~5 m in front of the camera (z=-4 looking +Z).
                b.center[0] = 0.0f;
                b.center[1] = 0.0f;
                b.center[2] = 5.0f;
                b.radius = 0.1f;
            } else {
                // Invisible: far outside any reasonable frustum on the +X side.
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
// Mode A — legacy v1 (no moments). pre_forward must stay on iota-all and never
// flip last_pre_forward_used_lru() to true.
// ============================================================================

TEST(TideStrategyLruTest, V1StoreFullCapacityDoesNotUseLruBranch) {
    SKIP_IF_NO_CUDA();
    TempDir tmp;
    const fs::path store_dir = tmp.path() / "store";
    constexpr std::size_t kNumBlocks = 3;
    make_synthetic_store(store_dir, kNumBlocks, /*with_moments=*/false);

    auto placeholder = make_placeholder();
    TideStrategy strategy(placeholder);
    auto opt = make_opt_params(store_dir, /*capacity=*/kNumBlocks);
    auto rt = lfs::training::tide::attach_tide_working_set(strategy, opt);
    ASSERT_TRUE(rt.has_value()) << rt.error();
    auto runtime = std::move(rt.value());
    strategy.initialize(opt);

    auto cam = make_camera_at(0.0f);
    ASSERT_NO_THROW(strategy.pre_forward(0, cam));

    EXPECT_FALSE(strategy.last_pre_forward_used_lru())
        << "v1 store must take the iota-all-blocks fast path; per-block Adam "
           "moments are required before variable residency is numerically safe.";
    EXPECT_TRUE(strategy.last_pre_forward_loaded());
    const auto& resident = strategy.last_resident_block_ids();
    ASSERT_EQ(resident.size(), kNumBlocks);
    for (std::size_t i = 0; i < kNumBlocks; ++i) {
        EXPECT_EQ(resident[i], i);
    }
}

// ============================================================================
// Mode B — v2 store that fits in WS capacity. Must still take iota-all path.
// ============================================================================

TEST(TideStrategyLruTest, V2StoreFitsInCapacityDoesNotUseLruBranch) {
    SKIP_IF_NO_CUDA();
    TempDir tmp;
    const fs::path store_dir = tmp.path() / "store";
    constexpr std::size_t kNumBlocks = 3;
    make_synthetic_store(store_dir, kNumBlocks, /*with_moments=*/true);

    auto placeholder = make_placeholder();
    TideStrategy strategy(placeholder);
    auto opt = make_opt_params(store_dir, /*capacity=*/kNumBlocks);
    auto rt = lfs::training::tide::attach_tide_working_set(strategy, opt);
    ASSERT_TRUE(rt.has_value()) << rt.error();
    auto runtime = std::move(rt.value());
    strategy.initialize(opt);

    auto cam = make_camera_at(0.0f);
    ASSERT_NO_THROW(strategy.pre_forward(0, cam));

    EXPECT_FALSE(strategy.last_pre_forward_used_lru())
        << "When num_blocks <= capacity no eviction is needed; iota-all is "
           "the correct fast path even with moments enabled.";
    const auto& resident = strategy.last_resident_block_ids();
    ASSERT_EQ(resident.size(), kNumBlocks);
    for (std::size_t i = 0; i < kNumBlocks; ++i) {
        EXPECT_EQ(resident[i], i);
    }
}

// ============================================================================
// Mode C — v2 store, num_blocks > capacity. The LRU+visibility policy must:
//   1) include every visible block in the resident set,
//   2) cap the resident set at exactly WS capacity,
//   3) sort the resident_scratch ascending (for deterministic equality-skip),
//   4) stamp block_last_used_iter == iter for all resident blocks.
// ============================================================================

TEST(TideStrategyLruTest, V2OverCapacityVisibleSubsetOfResidentCappedAtCapacity) {
    SKIP_IF_NO_CUDA();
    TempDir tmp;
    const fs::path store_dir = tmp.path() / "store";
    constexpr std::size_t kNumBlocks = 6;
    constexpr std::size_t kCapacity = 4; // strictly less than kNumBlocks
    make_synthetic_store(store_dir, kNumBlocks, /*with_moments=*/true);

    auto placeholder = make_placeholder();
    TideStrategy strategy(placeholder);
    auto opt = make_opt_params(store_dir, kCapacity);
    auto rt = lfs::training::tide::attach_tide_working_set(strategy, opt);
    ASSERT_TRUE(rt.has_value()) << rt.error();
    auto runtime = std::move(rt.value());
    strategy.initialize(opt);

    // Force exactly blocks {0, 1} into the camera frustum; |visible|=2 < capacity=4
    // ensures Mode C engages without tripping the "visible exceeds capacity" guard.
    force_visibility(*runtime.store, kNumBlocks, /*visible_ids=*/{0, 1});

    auto cam = make_camera_at(0.0f);
    ASSERT_NO_THROW(strategy.pre_forward(7, cam));

    EXPECT_TRUE(strategy.last_pre_forward_used_lru())
        << "moments_enabled && num_blocks > capacity must engage Mode C.";

    const auto& visible = strategy.last_visible_block_ids();
    const auto& resident = strategy.last_resident_block_ids();
    ASSERT_EQ(resident.size(), kCapacity);

    // Property 1: visible ⊆ resident.
    std::unordered_set<std::size_t> resident_set(resident.begin(), resident.end());
    for (auto vid : visible) {
        EXPECT_TRUE(resident_set.count(vid))
            << "Visible block " << vid << " missing from resident set "
            << "(visible-protection violated)";
    }

    // Property 2: resident is sorted ascending (for the equality-skip path).
    for (std::size_t i = 1; i < resident.size(); ++i) {
        EXPECT_LT(resident[i - 1], resident[i]) << "Resident set not sorted ascending";
    }

    // Property 3: every resident block has last_used_iter == 7.
    for (auto bid : resident) {
        EXPECT_EQ(strategy.block_last_used_iter(bid), std::int64_t{7})
            << "Resident block " << bid << " not stamped with current iter";
    }

    // Property 4: non-resident blocks were never made resident, lui stays -1.
    for (std::size_t bid = 0; bid < kNumBlocks; ++bid) {
        if (resident_set.count(bid)) continue;
        EXPECT_EQ(strategy.block_last_used_iter(bid), std::int64_t{-1})
            << "Non-resident block " << bid
            << " has lui != -1; was incorrectly stamped";
    }
}

// ============================================================================
// LRU semantics: across two iters with the same camera, the resident set must
// be stable (iter-0 fillers stay resident because they have the highest
// last_used_iter among non-visible candidates). This means iter-1 must SKIP
// the load (last_pre_forward_loaded() == false).
// ============================================================================

TEST(TideStrategyLruTest, SameCameraTwiceSkipsLoadAndKeepsResidentStable) {
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

    // Deterministic visibility: only blocks {0, 1} in-frustum; |visible|=2 < cap=4.
    force_visibility(*runtime.store, kNumBlocks, /*visible_ids=*/{0, 1});

    auto cam = make_camera_at(0.0f);

    // iter 0: cold load.
    strategy.pre_forward(0, cam);
    EXPECT_TRUE(strategy.last_pre_forward_loaded());
    const std::vector<std::size_t> resident_iter0 = strategy.last_resident_block_ids();
    ASSERT_EQ(resident_iter0.size(), kCapacity);

    // iter 1: same camera. Visible set unchanged. LRU fillers from iter 0
    // (lui=0) outrank never-resident blocks (lui=-1) so resident is identical.
    strategy.pre_forward(1, cam);
    EXPECT_FALSE(strategy.last_pre_forward_loaded())
        << "Same camera + LRU stability -> WorkingSet load_and_activate must be skipped.";
    const std::vector<std::size_t> resident_iter1 = strategy.last_resident_block_ids();
    EXPECT_EQ(resident_iter0, resident_iter1)
        << "LRU policy violated: resident set drifted across identical cameras";

    // iter-1 stamping: every still-resident block has lui=1, not 0.
    for (auto bid : resident_iter1) {
        EXPECT_EQ(strategy.block_last_used_iter(bid), std::int64_t{1});
    }
}

// ============================================================================
// LRU recency: when a previously-resident block falls out of visibility, it
// keeps a higher last_used_iter than a never-resident block, so it survives
// at least one more eviction round.
// ============================================================================

TEST(TideStrategyLruTest, LRUKeepsPreviouslyResidentOverNeverResident) {
    SKIP_IF_NO_CUDA();
    TempDir tmp;
    const fs::path store_dir = tmp.path() / "store";
    constexpr std::size_t kNumBlocks = 6;
    constexpr std::size_t kCapacity = 3;
    make_synthetic_store(store_dir, kNumBlocks, /*with_moments=*/true);

    auto placeholder = make_placeholder();
    TideStrategy strategy(placeholder);
    auto opt = make_opt_params(store_dir, kCapacity);
    auto rt = lfs::training::tide::attach_tide_working_set(strategy, opt);
    ASSERT_TRUE(rt.has_value()) << rt.error();
    auto runtime = std::move(rt.value());
    strategy.initialize(opt);

    // Deterministic visibility: only block {0} in-frustum; |visible|=1 < cap=3
    // leaves 2 LRU-fill slots that must come from the never-resident pool on
    // iter 0 (all lui=-1) and the stamped-iter-0 pool on iter 1 (lui=0 > -1).
    force_visibility(*runtime.store, kNumBlocks, /*visible_ids=*/{0});

    // Two iterations with the same camera. After each one, every resident
    // block_id has lui == that iter and every non-resident block_id has lui == -1.
    auto cam = make_camera_at(0.0f);
    strategy.pre_forward(0, cam);
    const std::vector<std::size_t> r0 = strategy.last_resident_block_ids();
    ASSERT_EQ(r0.size(), kCapacity);

    strategy.pre_forward(1, cam);
    const std::vector<std::size_t> r1 = strategy.last_resident_block_ids();

    // Core LRU contract: never-resident blocks have lui = -1 forever.
    // Any block in r1 must therefore appear in r0 (visible set is identical
    // across the two calls, and r0 already exhausted capacity).
    std::unordered_set<std::size_t> r0_set(r0.begin(), r0.end());
    for (auto bid : r1) {
        EXPECT_TRUE(r0_set.count(bid))
            << "Block " << bid << " entered resident set at iter 1 despite "
               "having lui=-1 vs iter-0 fillers with lui=0 — LRU policy lost.";
    }
    // And blocks that were never resident must still be at lui=-1.
    for (std::size_t bid = 0; bid < kNumBlocks; ++bid) {
        if (r0_set.count(bid)) continue;
        EXPECT_EQ(strategy.block_last_used_iter(bid), std::int64_t{-1});
    }
}
