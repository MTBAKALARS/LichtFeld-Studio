/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/camera_types.h"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "strategies/strategy_factory.hpp"
#include "strategies/tide_strategy.hpp"

#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
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
