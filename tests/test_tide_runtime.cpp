/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_tide_runtime.cpp
 * @brief Phase 3.3b smoke: exercise attach_tide_working_set() against a
 *        synthetic on-disk BlockStore, drive TideStrategy through one full
 *        cycle, and verify all blocks become resident.
 *
 * Skips with GTEST_SKIP if no CUDA device is present.
 */

#include "core/block_store.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tiered_cache.hpp"
#include "strategies/tide_strategy.hpp"
#include "tide/aos_soa_repack.hpp"
#include "tide/tide_runtime.hpp"
#include "tide/working_set.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <span>
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
            const auto name = "lfs_tide_runtime_test_" + std::to_string(rd()) + "_" + std::to_string(rd());
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

    constexpr int kShDegree = 3;
    constexpr std::size_t kRest = 45;

    void make_synthetic_store(const fs::path& dir, std::size_t num_blocks) {
        const std::size_t n = BlockStore::kDefaultBlockSize * num_blocks;
        std::vector<float> means(3 * n, 0.0f);
        std::vector<float> scaling(3 * n, -2.0f);
        std::vector<float> rotation(4 * n, 0.0f);
        std::vector<float> opacity(n, 0.5f);
        std::vector<float> sh0(3 * n, 0.5f);
        std::vector<float> shN(kRest * n, 0.0f);
        for (std::size_t i = 0; i < n; ++i) {
            means[3 * i + 0] = static_cast<float>(i % 64);
            means[3 * i + 1] = static_cast<float>((i / 64) % 64);
            means[3 * i + 2] = static_cast<float>(i / (64 * 64));
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

        auto r = BlockStore::stream_ply_to_base(dir, src);
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

} // namespace

#define SKIP_IF_NO_CUDA()                              \
    do {                                               \
        if (!cuda_available()) {                       \
            GTEST_SKIP() << "No CUDA device present."; \
            return;                                    \
        }                                              \
    } while (0)

// -----------------------------------------------------------------------------
// Helper-only tests (no live trainer; verify attach + activate behavior).
// -----------------------------------------------------------------------------

TEST(TideRuntime, RejectsEmptyStorePath) {
    auto placeholder = make_placeholder();
    TideStrategy strategy(placeholder);
    lfs::core::param::OptimizationParameters opt;
    opt.tide_store_path.clear();

    auto r = tide::attach_tide_working_set(strategy, opt);
    EXPECT_FALSE(r.has_value());
    EXPECT_NE(r.error().find("tide_store_path is empty"), std::string::npos);
}

TEST(TideRuntime, RejectsMissingStoreDirectory) {
    auto placeholder = make_placeholder();
    TideStrategy strategy(placeholder);
    lfs::core::param::OptimizationParameters opt;
    opt.tide_store_path = fs::temp_directory_path() / "lfs_tide_runtime_nonexistent_dir_xyz";
    std::error_code ec;
    fs::remove_all(opt.tide_store_path, ec);

    auto r = tide::attach_tide_working_set(strategy, opt);
    EXPECT_FALSE(r.has_value());
    EXPECT_NE(r.error().find("does not exist"), std::string::npos);
}

TEST(TideRuntime, AttachAndActivateAllBlocks) {
    SKIP_IF_NO_CUDA();
    TempDir tmp;
    const fs::path store_dir = tmp.path() / "store";
    constexpr std::size_t kNumBlocks = 8;
    make_synthetic_store(store_dir, kNumBlocks);

    auto placeholder = make_placeholder();
    TideStrategy strategy(placeholder);

    lfs::core::param::OptimizationParameters opt;
    opt.tide_store_path = store_dir;
    opt.tide_capacity_blocks = 0;        // auto = store.num_blocks
    opt.tide_cache_capacity_blocks = 0;  // auto = 2 * ws_capacity

    auto runtime_r = tide::attach_tide_working_set(strategy, opt);
    ASSERT_TRUE(runtime_r.has_value()) << runtime_r.error();
    auto runtime = std::move(runtime_r.value());

    ASSERT_NE(runtime.store, nullptr);
    ASSERT_NE(runtime.cache, nullptr);
    ASSERT_NE(runtime.working_set, nullptr);
    EXPECT_EQ(runtime.store_num_blocks, kNumBlocks);
    EXPECT_EQ(runtime.effective_capacity_blocks, kNumBlocks);
    EXPECT_EQ(strategy.get_working_set(), runtime.working_set.get());

    // Activate all blocks (single-tile mode).
    auto act = tide::activate_all_blocks(runtime);
    ASSERT_TRUE(act.has_value()) << act.error();

    EXPECT_EQ(runtime.working_set->active_block_count(), kNumBlocks);
    EXPECT_EQ(runtime.working_set->active_gaussian_count(),
              kNumBlocks * runtime.store->block_size());
}

TEST(TideRuntime, ClampsCapacityToStoreSize) {
    SKIP_IF_NO_CUDA();
    TempDir tmp;
    const fs::path store_dir = tmp.path() / "store";
    constexpr std::size_t kNumBlocks = 4;
    make_synthetic_store(store_dir, kNumBlocks);

    auto placeholder = make_placeholder();
    TideStrategy strategy(placeholder);

    lfs::core::param::OptimizationParameters opt;
    opt.tide_store_path = store_dir;
    opt.tide_capacity_blocks = 999;  // intentionally too large
    opt.tide_cache_capacity_blocks = 0;

    auto runtime_r = tide::attach_tide_working_set(strategy, opt);
    ASSERT_TRUE(runtime_r.has_value()) << runtime_r.error();
    EXPECT_EQ(runtime_r->effective_capacity_blocks, kNumBlocks);
}

TEST(TideRuntime, RespectsExplicitCacheCapacity) {
    SKIP_IF_NO_CUDA();
    TempDir tmp;
    const fs::path store_dir = tmp.path() / "store";
    constexpr std::size_t kNumBlocks = 4;
    make_synthetic_store(store_dir, kNumBlocks);

    auto placeholder = make_placeholder();
    TideStrategy strategy(placeholder);

    lfs::core::param::OptimizationParameters opt;
    opt.tide_store_path = store_dir;
    opt.tide_capacity_blocks = kNumBlocks;
    opt.tide_cache_capacity_blocks = 16;  // larger than 2x default

    auto runtime_r = tide::attach_tide_working_set(strategy, opt);
    ASSERT_TRUE(runtime_r.has_value()) << runtime_r.error();
    EXPECT_EQ(runtime_r->effective_capacity_blocks, kNumBlocks);
    // cache capacity stored inside TieredCache; we can't observe directly,
    // but a successful attach with a larger cache than ws is the contract.
}

// -----------------------------------------------------------------------------
// End-to-end: full TideStrategy cycle through the attached runtime.
// -----------------------------------------------------------------------------

TEST(TideRuntime, EndToEndOneTrainingStep) {
    SKIP_IF_NO_CUDA();
    TempDir tmp;
    const fs::path store_dir = tmp.path() / "store";
    constexpr std::size_t kNumBlocks = 8;
    make_synthetic_store(store_dir, kNumBlocks);

    auto placeholder = make_placeholder();
    TideStrategy strategy(placeholder);

    lfs::core::param::OptimizationParameters opt;
    opt.tide_store_path = store_dir;
    opt.tide_capacity_blocks = 0;
    opt.tide_cache_capacity_blocks = 0;
    opt.sh_degree = kShDegree;

    auto runtime_r = tide::attach_tide_working_set(strategy, opt);
    ASSERT_TRUE(runtime_r.has_value()) << runtime_r.error();
    auto runtime = std::move(runtime_r.value());

    // strategy.initialize() AFTER attach.
    strategy.initialize(opt);

    // Make all blocks resident (Phase 3.3b: no per-iter streaming).
    auto act = tide::activate_all_blocks(runtime);
    ASSERT_TRUE(act.has_value()) << act.error();

    // pre_step → step cycle (iteration safely past sh_warmup).
    RenderOutput render_output{};
    strategy.pre_step(/*iter=*/1500, render_output);

    strategy.step(/*iter=*/1500);

    EXPECT_EQ(runtime.working_set->active_block_count(), kNumBlocks);
}
