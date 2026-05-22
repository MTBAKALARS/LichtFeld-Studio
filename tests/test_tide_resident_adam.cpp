/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_tide_resident_adam.cpp
 * @brief Phase 3.1 tests for tide::TideResidentAdam.
 *
 * Requires a live CUDA context. Skips with GTEST_SKIP if cudaSetDevice fails
 * (e.g. CI box with no GPU). Verifies:
 *   - Numerical equivalence to a pure CPU Adam reference implementation for
 *     a representative ParamType (Means) over multiple iterations.
 *   - Per-ParamType step_count bookkeeping is independent across types.
 *   - SH warmup gate (iteration <= sh_warmup_iterations) skips ShN entirely.
 *   - reset() clears state for one ParamType without touching siblings.
 *   - Mismatched num_elements between create() and step() is rejected.
 */

#include "optimizer/adam_optimizer.hpp" // AdamConfig, ParamType
#include "optimizer/tide_resident_adam.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace {

    using lfs::training::AdamConfig;
    using lfs::training::ParamType;
    using lfs::training::tide::TideResidentAdam;

    constexpr int kCudaDevice = 0;

    // Test fixture: skip the whole suite if no CUDA device is available.
    class TideResidentAdamTest : public ::testing::Test {
    protected:
        void SetUp() override {
            if (cudaSetDevice(kCudaDevice) != cudaSuccess) {
                GTEST_SKIP() << "No CUDA device " << kCudaDevice << " available";
            }
        }
    };

    // Allocate a device buffer and copy host data into it. Caller frees with cudaFree.
    float* device_upload(const std::vector<float>& host) {
        float* d = nullptr;
        const std::size_t bytes = host.size() * sizeof(float);
        EXPECT_EQ(cudaMalloc(reinterpret_cast<void**>(&d), bytes), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d, host.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
        return d;
    }

    std::vector<float> device_download(const float* device, std::size_t n) {
        std::vector<float> host(n);
        EXPECT_EQ(cudaMemcpy(host.data(), device, n * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        return host;
    }

    // Reference CPU Adam step matching adam_step_raw semantics:
    //   m = beta1 * m + (1 - beta1) * g
    //   v = beta2 * v + (1 - beta2) * g * g
    //   m_hat = m * bc1_rcp        (bc1_rcp = 1 / (1 - beta1^step))
    //   v_hat = v * (bc2_sqrt_rcp)^2  (bc2_sqrt_rcp = 1 / sqrt(1 - beta2^step))
    //   param -= lr * m_hat / (sqrt(v_hat) + eps)
    void cpu_adam_step(std::vector<float>& param,
                       std::vector<float>& exp_avg,
                       std::vector<float>& exp_avg_sq,
                       const std::vector<float>& grad,
                       float lr, float beta1, float beta2, float eps,
                       int step) {
        const double bc1_rcp = 1.0 / (1.0 - std::pow(static_cast<double>(beta1), step));
        const double bc2_sqrt_rcp = 1.0 / std::sqrt(
            1.0 - std::pow(static_cast<double>(beta2), step));
        for (std::size_t i = 0; i < param.size(); ++i) {
            const float g = grad[i];
            exp_avg[i] = beta1 * exp_avg[i] + (1.0f - beta1) * g;
            exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g * g;
            const float m_hat = exp_avg[i] * static_cast<float>(bc1_rcp);
            const float v_hat = exp_avg_sq[i] *
                                static_cast<float>(bc2_sqrt_rcp * bc2_sqrt_rcp);
            param[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }

    void expect_close(const std::vector<float>& a, const std::vector<float>& b,
                      float abs_tol, float rel_tol) {
        ASSERT_EQ(a.size(), b.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            const float diff = std::abs(a[i] - b[i]);
            const float allowed = abs_tol + rel_tol * std::max(std::abs(a[i]),
                                                                std::abs(b[i]));
            EXPECT_LE(diff, allowed)
                << "  index=" << i << "  a=" << a[i] << "  b=" << b[i];
        }
    }

    // -------------------------------------------------------------
    // Test 1: numerical equivalence to a CPU Adam reference.
    // -------------------------------------------------------------
    TEST_F(TideResidentAdamTest, MatchesCpuAdamReferenceOverManySteps) {
        constexpr std::size_t kNumGaussians = 128;
        constexpr std::size_t kFeatureDim = 3;
        constexpr std::size_t kN = kNumGaussians * kFeatureDim;
        constexpr int kIters = 25;

        // Seeded random init.
        std::mt19937 rng(0xC0FFEE);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> param_host(kN);
        std::vector<float> grad_host(kN);
        for (auto& v : param_host) v = dist(rng);
        for (auto& v : grad_host) v = dist(rng) * 0.1f;

        // CPU reference state.
        std::vector<float> param_ref = param_host;
        std::vector<float> exp_avg_ref(kN, 0.0f);
        std::vector<float> exp_avg_sq_ref(kN, 0.0f);

        // GPU device-under-test.
        AdamConfig adam_cfg{};
        adam_cfg.lr = 1e-3f;
        adam_cfg.beta1 = 0.9;
        adam_cfg.beta2 = 0.999;
        adam_cfg.eps = 1e-15;

        TideResidentAdam::Config cfg{};
        cfg.adam = adam_cfg;
        cfg.cuda_device = kCudaDevice;

        std::array<TideResidentAdam::ParamSpec, 1> specs{{{ParamType::Means, kN}}};
        auto opt_or = TideResidentAdam::create(cfg, specs);
        ASSERT_TRUE(opt_or.has_value()) << opt_or.error();
        auto& opt = *opt_or.value();

        float* d_param = device_upload(param_host);
        float* d_grad = device_upload(grad_host);

        for (int iter = 1; iter <= kIters; ++iter) {
            // Run GPU step.
            auto r = opt.step(ParamType::Means, d_param, d_grad, kN, iter);
            ASSERT_TRUE(r.has_value()) << r.error();

            // Mirror with CPU reference.
            cpu_adam_step(param_ref, exp_avg_ref, exp_avg_sq_ref, grad_host,
                          adam_cfg.lr, static_cast<float>(adam_cfg.beta1),
                          static_cast<float>(adam_cfg.beta2),
                          static_cast<float>(adam_cfg.eps), iter);
        }

        EXPECT_EQ(opt.step_count(ParamType::Means), kIters);

        auto param_gpu = device_download(d_param, kN);
        auto exp_avg_gpu = device_download(opt.exp_avg(ParamType::Means), kN);
        auto exp_avg_sq_gpu = device_download(opt.exp_avg_sq(ParamType::Means), kN);

        // Very tight tolerances — the only divergence is float-order-of-ops
        // between the CUDA kernel and our CPU loop.
        expect_close(param_gpu, param_ref, 1e-6f, 1e-5f);
        expect_close(exp_avg_gpu, exp_avg_ref, 1e-7f, 1e-6f);
        expect_close(exp_avg_sq_gpu, exp_avg_sq_ref, 1e-9f, 1e-6f);

        cudaFree(d_param);
        cudaFree(d_grad);
    }

    // -------------------------------------------------------------
    // Test 2: per-ParamType step_count is independent.
    // -------------------------------------------------------------
    TEST_F(TideResidentAdamTest, StepCountIsPerParamType) {
        constexpr std::size_t kN = 64;

        AdamConfig adam_cfg{};
        adam_cfg.lr = 1e-3f;

        TideResidentAdam::Config cfg{};
        cfg.adam = adam_cfg;
        cfg.cuda_device = kCudaDevice;

        std::array<TideResidentAdam::ParamSpec, 2> specs{{
            {ParamType::Means, kN},
            {ParamType::Opacity, kN},
        }};
        auto opt_or = TideResidentAdam::create(cfg, specs);
        ASSERT_TRUE(opt_or.has_value()) << opt_or.error();
        auto& opt = *opt_or.value();

        std::vector<float> param_host(kN, 0.5f);
        std::vector<float> grad_host(kN, 0.01f);
        float* d_param = device_upload(param_host);
        float* d_grad = device_upload(grad_host);

        for (int i = 1; i <= 3; ++i) {
            ASSERT_TRUE(opt.step(ParamType::Means, d_param, d_grad, kN, i).has_value());
        }
        for (int i = 1; i <= 5; ++i) {
            ASSERT_TRUE(opt.step(ParamType::Opacity, d_param, d_grad, kN, i).has_value());
        }

        EXPECT_EQ(opt.step_count(ParamType::Means), 3);
        EXPECT_EQ(opt.step_count(ParamType::Opacity), 5);
        // Unregistered types should remain at 0.
        EXPECT_EQ(opt.step_count(ParamType::Rotation), 0);
        EXPECT_FALSE(opt.has_param(ParamType::Rotation));

        cudaFree(d_param);
        cudaFree(d_grad);
    }

    // -------------------------------------------------------------
    // Test 3: SH warmup gate skips ShN param updates.
    // -------------------------------------------------------------
    TEST_F(TideResidentAdamTest, ShWarmupSkipsShNUpdates) {
        constexpr std::size_t kN = 32;

        AdamConfig adam_cfg{};
        adam_cfg.lr = 1e-3f;

        TideResidentAdam::Config cfg{};
        cfg.adam = adam_cfg;
        cfg.cuda_device = kCudaDevice;
        cfg.sh_warmup_iterations = 100;

        std::array<TideResidentAdam::ParamSpec, 1> specs{{{ParamType::ShN, kN}}};
        auto opt_or = TideResidentAdam::create(cfg, specs);
        ASSERT_TRUE(opt_or.has_value()) << opt_or.error();
        auto& opt = *opt_or.value();

        std::vector<float> param_host(kN, 2.0f);
        std::vector<float> grad_host(kN, 1.0f);
        float* d_param = device_upload(param_host);
        float* d_grad = device_upload(grad_host);

        // Steps inside the warmup window must NOT modify params.
        for (int iter = 1; iter <= 100; ++iter) {
            ASSERT_TRUE(opt.step(ParamType::ShN, d_param, d_grad, kN, iter).has_value());
        }
        auto after_warmup = device_download(d_param, kN);
        for (auto v : after_warmup) {
            EXPECT_FLOAT_EQ(v, 2.0f) << "ShN param modified during warmup";
        }
        // But step_count still advanced (mirrors AdamOptimizer behavior).
        EXPECT_EQ(opt.step_count(ParamType::ShN), 100);
        EXPECT_EQ(opt.stats().steps_skipped, 100u);
        EXPECT_EQ(opt.stats().steps_total, 0u);

        // First post-warmup step must modify params.
        ASSERT_TRUE(opt.step(ParamType::ShN, d_param, d_grad, kN, 101).has_value());
        auto after_step = device_download(d_param, kN);
        for (auto v : after_step) {
            EXPECT_NE(v, 2.0f) << "ShN param NOT modified after warmup";
        }
        EXPECT_EQ(opt.stats().steps_total, 1u);

        cudaFree(d_param);
        cudaFree(d_grad);
    }

    // -------------------------------------------------------------
    // Test 4: reset() clears state for one type only.
    // -------------------------------------------------------------
    TEST_F(TideResidentAdamTest, ResetClearsStateForOneTypeOnly) {
        constexpr std::size_t kN = 16;

        AdamConfig adam_cfg{};
        adam_cfg.lr = 1e-3f;

        TideResidentAdam::Config cfg{};
        cfg.adam = adam_cfg;
        cfg.cuda_device = kCudaDevice;

        std::array<TideResidentAdam::ParamSpec, 2> specs{{
            {ParamType::Means, kN},
            {ParamType::Opacity, kN},
        }};
        auto opt_or = TideResidentAdam::create(cfg, specs);
        ASSERT_TRUE(opt_or.has_value()) << opt_or.error();
        auto& opt = *opt_or.value();

        std::vector<float> p(kN, 0.5f);
        std::vector<float> g(kN, 0.1f);
        float* dp = device_upload(p);
        float* dg = device_upload(g);

        for (int i = 1; i <= 4; ++i) {
            ASSERT_TRUE(opt.step(ParamType::Means, dp, dg, kN, i).has_value());
            ASSERT_TRUE(opt.step(ParamType::Opacity, dp, dg, kN, i).has_value());
        }

        // exp_avg should be non-zero for both types after stepping.
        auto means_m_before = device_download(opt.exp_avg(ParamType::Means), kN);
        auto opacity_m_before = device_download(opt.exp_avg(ParamType::Opacity), kN);
        EXPECT_GT(std::abs(means_m_before[0]), 0.0f);
        EXPECT_GT(std::abs(opacity_m_before[0]), 0.0f);

        ASSERT_TRUE(opt.reset(ParamType::Means).has_value());

        EXPECT_EQ(opt.step_count(ParamType::Means), 0);
        EXPECT_EQ(opt.step_count(ParamType::Opacity), 4);

        auto means_m_after = device_download(opt.exp_avg(ParamType::Means), kN);
        auto opacity_m_after = device_download(opt.exp_avg(ParamType::Opacity), kN);
        for (auto v : means_m_after) EXPECT_FLOAT_EQ(v, 0.0f);
        for (std::size_t i = 0; i < kN; ++i) {
            EXPECT_FLOAT_EQ(opacity_m_after[i], opacity_m_before[i])
                << "Opacity state corrupted by Means reset, idx=" << i;
        }

        cudaFree(dp);
        cudaFree(dg);
    }

    // -------------------------------------------------------------
    // Test 5: mismatched num_elements is rejected with a clean error.
    // -------------------------------------------------------------
    TEST_F(TideResidentAdamTest, MismatchedNumElementsIsRejected) {
        constexpr std::size_t kN = 64;

        AdamConfig adam_cfg{};
        adam_cfg.lr = 1e-3f;

        TideResidentAdam::Config cfg{};
        cfg.adam = adam_cfg;
        cfg.cuda_device = kCudaDevice;

        std::array<TideResidentAdam::ParamSpec, 1> specs{{{ParamType::Means, kN}}};
        auto opt_or = TideResidentAdam::create(cfg, specs);
        ASSERT_TRUE(opt_or.has_value()) << opt_or.error();
        auto& opt = *opt_or.value();

        std::vector<float> p(kN, 0.0f);
        std::vector<float> g(kN, 0.0f);
        float* dp = device_upload(p);
        float* dg = device_upload(g);

        auto r = opt.step(ParamType::Means, dp, dg, kN + 1, 1);
        EXPECT_FALSE(r.has_value());
        EXPECT_NE(r.error().find("num_elements mismatch"), std::string::npos);

        // step_count must NOT advance on a rejected call.
        EXPECT_EQ(opt.step_count(ParamType::Means), 0);

        // Unregistered type is also rejected.
        auto r2 = opt.step(ParamType::Opacity, dp, dg, kN, 1);
        EXPECT_FALSE(r2.has_value());
        EXPECT_NE(r2.error().find("not registered"), std::string::npos);

        cudaFree(dp);
        cudaFree(dg);
    }

} // namespace
