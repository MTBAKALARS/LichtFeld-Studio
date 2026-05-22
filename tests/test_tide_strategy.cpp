/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/block_store.hpp"
#include "core/camera.hpp"
#include "core/camera_types.h"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "strategies/strategy_factory.hpp"
#include "strategies/tide_strategy.hpp"
#include "tide/frustum_culler.hpp"
#include "tide/tide_runtime.hpp"

#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    bool has_cuda_device() {
        int count = 0;
        const cudaError_t err = cudaGetDeviceCount(&count);
        return err == cudaSuccess && count > 0;
    }

    SplatData make_test_splat_data(int n_gaussians = 64, int sh_degree = 3) {
        const std::size_t n = static_cast<std::size_t>(n_gaussians);
        const std::size_t sh_rest = (sh_degree <= 0)
                                        ? 0u
                                        : static_cast<std::size_t>((sh_degree + 1) * (sh_degree + 1) - 1);

        std::vector<float> means_data(n * 3, 0.0f);
        std::vector<float> sh0_data(n * 3, 0.5f);
        std::vector<float> shN_data(n * sh_rest * 3, 0.0f);
        std::vector<float> scaling_data(n * 3, -2.0f);
        std::vector<float> rotation_data(n * 4, 0.0f);
        std::vector<float> opacity_data(n * 1, 0.5f);

        for (std::size_t i = 0; i < n; ++i) {
            rotation_data[i * 4 + 0] = 1.0f; // identity quaternion
            means_data[i * 3 + 0] = static_cast<float>(i) * 0.01f;
        }

        auto means = Tensor::from_vector(means_data, TensorShape({n, 3}), Device::CUDA);
        auto sh0 = Tensor::from_vector(sh0_data, TensorShape({n, 1, 3}), Device::CUDA);
        auto shN = (sh_rest == 0)
                       ? Tensor::zeros({n, 0, 3}, Device::CUDA)
                       : Tensor::from_vector(shN_data, TensorShape({n, sh_rest, 3}), Device::CUDA);
        auto scaling = Tensor::from_vector(scaling_data, TensorShape({n, 3}), Device::CUDA);
        auto rotation = Tensor::from_vector(rotation_data, TensorShape({n, 4}), Device::CUDA);
        auto opacity = Tensor::from_vector(opacity_data, TensorShape({n, 1}), Device::CUDA);

        return SplatData(sh_degree, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    }

    param::OptimizationParameters make_opt_params(int iterations = 100) {
        param::OptimizationParameters p;
        p.iterations = static_cast<size_t>(iterations);
        p.max_cap = 256;
        return p;
    }

    Camera make_test_camera() {
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
                      /*image_name=*/"test_cam",
                      /*image_path=*/"",
                      /*mask_path=*/std::filesystem::path{},
                      /*camera_width=*/640, /*camera_height=*/480,
                      /*uid=*/0);
    }

} // namespace

#define SKIP_IF_NO_CUDA()                              \
    do {                                               \
        if (!has_cuda_device()) {                      \
            GTEST_SKIP() << "No CUDA device present."; \
        }                                              \
    } while (0)

TEST(TideStrategyTest, ConstructWithoutInitializeIsCheap) {
    SKIP_IF_NO_CUDA();
    auto splat = make_test_splat_data();
    TideStrategy strategy(splat);
    EXPECT_STREQ(strategy.strategy_type(), "tide");
    EXPECT_FALSE(strategy.is_refining(0));
    EXPECT_EQ(strategy.soa_capacity_gaussians(), 0u);
    EXPECT_EQ(strategy.soa_scratch_bytes(), 0u);
    EXPECT_EQ(strategy.get_resident_adam(), nullptr);
}

TEST(TideStrategyTest, InitializeAllocatesSoaScratchAndExposesView) {
    SKIP_IF_NO_CUDA();
    constexpr int kN = 64;
    constexpr int kSh = 3;
    auto splat = make_test_splat_data(kN, kSh);
    TideStrategy strategy(splat);
    strategy.initialize(make_opt_params());

    EXPECT_EQ(strategy.soa_capacity_gaussians(), static_cast<std::size_t>(kN));
    // Expected: (3 + 3 + 4 + 1 + 3 + ((kSh+1)^2-1)*3) floats per Gaussian.
    constexpr std::size_t kExpectedFloatsPerG = 3 + 3 + 4 + 1 + 3 + ((kSh + 1) * (kSh + 1) - 1) * 3;
    EXPECT_EQ(strategy.soa_scratch_bytes(), sizeof(float) * kN * kExpectedFloatsPerG);

    auto& model = strategy.get_model();
    EXPECT_EQ(model.size(), kN);
    EXPECT_EQ(model.get_max_sh_degree(), kSh);
    EXPECT_EQ(model.means_raw().shape()[0], static_cast<std::size_t>(kN));
    EXPECT_EQ(model.means_raw().shape()[1], 3u);
    EXPECT_EQ(model.scaling_raw().shape()[1], 3u);
    EXPECT_EQ(model.rotation_raw().shape()[1], 4u);
    EXPECT_EQ(model.opacity_raw().shape()[1], 1u);

    // SOA buffer must NOT alias the placeholder buffer — the view is backed
    // by our own cudaMalloc.
    EXPECT_NE(model.means_raw().ptr<float>(), splat.means_raw().ptr<float>());

    auto& opt = strategy.get_optimizer();
    (void)opt; // simply must not throw
    EXPECT_NE(strategy.get_resident_adam(), nullptr);
}

TEST(TideStrategyTest, BootstrapCopiesPlaceholderData) {
    SKIP_IF_NO_CUDA();
    constexpr int kN = 32;
    auto splat = make_test_splat_data(kN, 1);
    TideStrategy strategy(splat);
    strategy.initialize(make_opt_params());

    // Means were initialized to i * 0.01f along x; sample a few values.
    const auto host_means = strategy.get_model().means_raw().cpu().to_vector();
    ASSERT_EQ(host_means.size(), static_cast<std::size_t>(kN * 3));
    EXPECT_FLOAT_EQ(host_means[0], 0.0f);
    EXPECT_FLOAT_EQ(host_means[3], 0.01f);
    EXPECT_FLOAT_EQ(host_means[3 * (kN - 1)], static_cast<float>(kN - 1) * 0.01f);

    // sh0 was filled with 0.5f.
    const auto host_sh0 = strategy.get_model().sh0_raw().cpu().to_vector();
    ASSERT_FALSE(host_sh0.empty());
    EXPECT_FLOAT_EQ(host_sh0[0], 0.5f);
}

TEST(TideStrategyTest, StepBeforeIterationsRunsCleanly) {
    SKIP_IF_NO_CUDA();
    auto splat = make_test_splat_data(64, 3);
    TideStrategy strategy(splat);
    strategy.initialize(make_opt_params(/*iterations=*/50));
    // Should be a no-op (no gradients populated), but must not throw.
    strategy.step(0);
    strategy.step(1);
}

TEST(TideStrategyTest, RemoveGaussiansIsSoftIgnore) {
    SKIP_IF_NO_CUDA();
    auto splat = make_test_splat_data();
    TideStrategy strategy(splat);
    strategy.initialize(make_opt_params());
    // Mask shape doesn't matter — Phase 3.2b ignores the call.
    auto mask = Tensor::zeros_bool({static_cast<std::size_t>(splat.size())}, Device::CUDA);
    strategy.remove_gaussians(mask);
    // Sanity: model still has the same number of Gaussians.
    EXPECT_EQ(strategy.get_model().size(), splat.size());
}

TEST(TideStrategyTest, SerializeDeserializeRoundtripsHeader) {
    SKIP_IF_NO_CUDA();
    auto splat = make_test_splat_data();
    TideStrategy strategy(splat);
    strategy.initialize(make_opt_params());

    std::stringstream ss;
    strategy.serialize(ss);
    EXPECT_GT(ss.tellp(), std::streampos(0));

    // Build a second strategy and deserialize.
    auto splat2 = make_test_splat_data();
    TideStrategy strategy2(splat2);
    strategy2.initialize(make_opt_params());
    EXPECT_NO_THROW(strategy2.deserialize(ss));
}

TEST(TideStrategyTest, RegisteredInFactory) {
    SKIP_IF_NO_CUDA();
    auto& factory = StrategyFactory::instance();
    EXPECT_TRUE(factory.has(std::string(param::kStrategyTide)));

    auto splat = make_test_splat_data();
    auto created = factory.create(std::string(param::kStrategyTide), splat);
    ASSERT_TRUE(created.has_value()) << (created ? "" : created.error());
    ASSERT_NE(created.value(), nullptr);
    EXPECT_STREQ(created.value()->strategy_type(), "tide");
}

// --- Phase 3.5.1: IStrategy::pre_forward callback ------------------------
//
// 3.5.1 wires the new per-iteration camera-aware hook into IStrategy and
// TideStrategy. The body of TideStrategy::pre_forward is intentionally empty
// in 3.5.1 (frustum-driven block activation lands in 3.5.2); these tests
// pin the contract that the callback exists, can be invoked safely many
// times, dispatches polymorphically through IStrategy, and does not perturb
// strategy state.

TEST(TideStrategyTest, PreForwardIsCallableBeforeInitialize) {
    SKIP_IF_NO_CUDA();
    auto splat = make_test_splat_data();
    TideStrategy strategy(splat);
    auto cam = make_test_camera();
    // Calling pre_forward on a not-yet-initialized strategy must not crash.
    // 3.5.2 will guard against use-before-initialize internally; in 3.5.1
    // the body is empty so any inputs are valid.
    EXPECT_NO_THROW(strategy.pre_forward(0, cam));
    EXPECT_EQ(strategy.soa_capacity_gaussians(), 0u);
}

TEST(TideStrategyTest, PreForwardIsNoOpAfterInitializeWithoutWorkingSet) {
    SKIP_IF_NO_CUDA();
    constexpr int kN = 32;
    auto splat = make_test_splat_data(kN, /*sh_degree=*/2);
    TideStrategy strategy(splat);
    strategy.initialize(make_opt_params());

    const std::size_t cap_before = strategy.soa_capacity_gaussians();
    const std::size_t bytes_before = strategy.soa_scratch_bytes();
    ASSERT_GT(cap_before, 0u);

    auto cam = make_test_camera();
    // Call many times across a synthetic iteration range; nothing should
    // change about the strategy's resident state.
    for (int iter = 0; iter < 16; ++iter) {
        ASSERT_NO_THROW(strategy.pre_forward(iter, cam));
    }
    EXPECT_EQ(strategy.soa_capacity_gaussians(), cap_before);
    EXPECT_EQ(strategy.soa_scratch_bytes(), bytes_before);
    EXPECT_EQ(strategy.get_working_set(), nullptr);

    // Phase 3.5.2: without sources attached, telemetry stays empty/false.
    EXPECT_EQ(strategy.last_visible_block_count(), 0u);
    EXPECT_TRUE(strategy.last_visible_block_ids().empty());
    EXPECT_FALSE(strategy.last_pre_forward_loaded());
}

TEST(TideStrategyTest, PreForwardDispatchesViaIStrategyBasePointer) {
    SKIP_IF_NO_CUDA();
    auto splat = make_test_splat_data();
    auto strategy = std::make_unique<TideStrategy>(splat);
    strategy->initialize(make_opt_params());

    IStrategy* base = strategy.get();
    auto cam = make_test_camera();
    // Virtual dispatch must reach TideStrategy::pre_forward without throwing
    // even though the default IStrategy::pre_forward is a no-op too — this
    // pins the override is wired correctly.
    EXPECT_NO_THROW(base->pre_forward(0, cam));
    EXPECT_NO_THROW(base->pre_forward(42, cam));
}


// ============================================================================
// Phase 3.5.2 — frustum-driven residency tests.
// ============================================================================

namespace {

    namespace fs = std::filesystem;

    class TideTempDir {
    public:
        TideTempDir() {
            std::random_device rd;
            const auto name = "lfs_tide_strategy_test_" + std::to_string(rd()) + "_" + std::to_string(rd());
            path_ = fs::temp_directory_path() / name;
            fs::create_directories(path_);
        }
        ~TideTempDir() {
            std::error_code ec;
            fs::remove_all(path_, ec);
        }
        TideTempDir(const TideTempDir&) = delete;
        TideTempDir& operator=(const TideTempDir&) = delete;
        const fs::path& path() const { return path_; }
    private:
        fs::path path_;
    };

    // Build a tiny on-disk BlockStore with `num_blocks` blocks where each
    // block's spatial bounds end up centered along the X-axis. Means are
    // arranged in a regular 64-per-side cube within each block, scaled so
    // that consecutive blocks land far apart (block i centered roughly at
    // x = i * 1000) — this gives frustum culling something meaningful to
    // distinguish.
    void make_synthetic_tide_store(const fs::path& dir, std::size_t num_blocks) {
        constexpr std::size_t kRest = 45;
        const std::size_t n = BlockStore::kDefaultBlockSize * num_blocks;
        std::vector<float> means(3 * n, 0.0f);
        std::vector<float> scaling(3 * n, -2.0f);
        std::vector<float> rotation(4 * n, 0.0f);
        std::vector<float> opacity(n, 0.5f);
        std::vector<float> sh0(3 * n, 0.5f);
        std::vector<float> shN(kRest * n, 0.0f);
        for (std::size_t i = 0; i < n; ++i) {
            // Each block is 4096 Gaussians; spread them across the X axis
            // so the streaming Morton order puts neighbors in the same block.
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

        auto r = BlockStore::stream_ply_to_base(dir, src);
        ASSERT_TRUE(r.has_value()) << "stream_ply_to_base: " << r.error();
    }

} // namespace

TEST(TideFrustumPlanesTest, IdentityCameraProducesNormalizedPlanes) {
    SKIP_IF_NO_CUDA();
    auto cam = make_test_camera();
    std::array<lfs::training::tide::FrustumCuller::Plane, 6> planes{};
    ASSERT_NO_THROW(
        lfs::training::tide::FrustumCuller::compute_frustum_planes(
            cam, /*near=*/0.01f, /*far=*/1000.0f, planes));

    for (std::size_t k = 0; k < 6; ++k) {
        const auto& p = planes[k];
        const float n2 = p.a * p.a + p.b * p.b + p.c * p.c;
        EXPECT_NEAR(n2, 1.0f, 1e-4f)
            << "Plane " << k << " is not unit-normalized: (a,b,c)=("
            << p.a << "," << p.b << "," << p.c << ")";
    }
}

TEST(TideFrustumPlanesTest, RejectsBadNearFar) {
    SKIP_IF_NO_CUDA();
    auto cam = make_test_camera();
    std::array<lfs::training::tide::FrustumCuller::Plane, 6> planes{};
    EXPECT_THROW(
        lfs::training::tide::FrustumCuller::compute_frustum_planes(cam, 0.0f, 1.0f, planes),
        std::runtime_error);
    EXPECT_THROW(
        lfs::training::tide::FrustumCuller::compute_frustum_planes(cam, 1.0f, 1.0f, planes),
        std::runtime_error);
    EXPECT_THROW(
        lfs::training::tide::FrustumCuller::compute_frustum_planes(cam, 10.0f, 1.0f, planes),
        std::runtime_error);
}

TEST(TideStrategyPreForwardTest, NoSourcesAttachedRemainsNoOp) {
    SKIP_IF_NO_CUDA();
    auto splat = make_test_splat_data();
    TideStrategy strategy(splat);
    strategy.initialize(make_opt_params());

    // set_tide_sources with null pointers must NOT enable the real path:
    // both store and cache need to be non-null.
    strategy.set_tide_sources(nullptr, nullptr);

    auto cam = make_test_camera();
    ASSERT_NO_THROW(strategy.pre_forward(0, cam));
    EXPECT_FALSE(strategy.last_pre_forward_loaded());
    EXPECT_EQ(strategy.last_visible_block_count(), 0u);
}

TEST(TideStrategyPreForwardTest, FullResidencyEndToEnd) {
    SKIP_IF_NO_CUDA();
    TideTempDir tmp;
    const fs::path store_dir = tmp.path() / "store";
    constexpr std::size_t kNumBlocks = 4;
    make_synthetic_tide_store(store_dir, kNumBlocks);

    // Build a runtime via attach_tide_working_set, which wires both
    // set_working_set AND set_tide_sources for us.
    auto placeholder = make_test_splat_data(/*n=*/4, /*sh_degree=*/3);
    TideStrategy strategy(placeholder);
    lfs::core::param::OptimizationParameters opt = make_opt_params();
    opt.tide_store_path = store_dir;
    opt.tide_capacity_blocks = kNumBlocks;  // full residency

    auto rt_result = lfs::training::tide::attach_tide_working_set(strategy, opt);
    ASSERT_TRUE(rt_result.has_value()) << rt_result.error();
    auto runtime = std::move(rt_result.value());

    strategy.initialize(opt);
    ASSERT_NE(strategy.get_working_set(), nullptr);

    auto cam = make_test_camera();

    // First call: should compute visible_ids AND issue a load_and_activate.
    ASSERT_NO_THROW(strategy.pre_forward(0, cam));
    EXPECT_TRUE(strategy.last_pre_forward_loaded())
        << "First pre_forward must populate the WorkingSet via load_and_activate";
    EXPECT_GT(strategy.last_visible_block_count(), 0u)
        << "FrustumCuller should report at least one visible block for an "
           "identity-rotation camera centered at origin";
    EXPECT_LE(strategy.last_visible_block_count(), kNumBlocks);

    // Verify the WorkingSet was actually populated.
    auto* ws = strategy.get_working_set();
    ASSERT_NE(ws, nullptr);
    EXPECT_EQ(ws->active_block_count(), kNumBlocks)
        << "Phase 3.5.2 deliberately forces full residency until per-block "
           "Adam moments (3.5.3) land";

    // Second call with the same camera: resident set is unchanged, so the
    // strategy must SKIP the load (perf parity with Phase 3.4c).
    ASSERT_NO_THROW(strategy.pre_forward(1, cam));
    EXPECT_FALSE(strategy.last_pre_forward_loaded())
        << "Idempotent re-load must be skipped when the resident set is unchanged";
    EXPECT_GT(strategy.last_visible_block_count(), 0u)
        << "Visible-id telemetry must still be computed even when the load is skipped";

    // WorkingSet is still active.
    EXPECT_EQ(ws->active_block_count(), kNumBlocks);
}

TEST(TideStrategyPreForwardTest, VisibleIdsAreSortedAscending) {
    SKIP_IF_NO_CUDA();
    TideTempDir tmp;
    const fs::path store_dir = tmp.path() / "store";
    constexpr std::size_t kNumBlocks = 4;
    make_synthetic_tide_store(store_dir, kNumBlocks);

    auto placeholder = make_test_splat_data(/*n=*/4, /*sh_degree=*/3);
    TideStrategy strategy(placeholder);
    lfs::core::param::OptimizationParameters opt = make_opt_params();
    opt.tide_store_path = store_dir;
    opt.tide_capacity_blocks = kNumBlocks;

    auto rt_result = lfs::training::tide::attach_tide_working_set(strategy, opt);
    ASSERT_TRUE(rt_result.has_value()) << rt_result.error();
    auto runtime = std::move(rt_result.value());

    strategy.initialize(opt);
    auto cam = make_test_camera();
    strategy.pre_forward(0, cam);

    const auto& ids = strategy.last_visible_block_ids();
    for (std::size_t i = 1; i < ids.size(); ++i) {
        EXPECT_LT(ids[i - 1], ids[i])
            << "FrustumCuller::cull guarantees ascending block_id order";
    }
}
