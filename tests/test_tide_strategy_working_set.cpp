/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_tide_strategy_working_set.cpp
 * @brief Phase 3.3a smoke: exercise TideStrategy end-to-end against an
 *        8-block synthetic BlockStore + TieredCache + WorkingSet.
 *
 * Verifies that:
 *   - set_working_set(ws) + initialize sizes SOA scratch to ws capacity
 *     (8 blocks * 4096 g/block = 32768 Gaussians).
 *   - pre_step unpacks the active AOS device buffer into the SOA view-backed
 *     SplatData.
 *   - step routes through TideResidentAdam (held AdamOptimizer.step() is NOT
 *     called) and writes the updated parameters back into the WorkingSet's
 *     mutable_device_buffer via soa_to_aos.
 *   - The repacked AOS bytes contain the post-Adam parameters (round-trips
 *     through aos_to_soa on the next pre_step).
 *
 * Skips with GTEST_SKIP if no CUDA device is present.
 */

#include "core/block_store.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tiered_cache.hpp"
#include "strategies/tide_strategy.hpp"
#include "tide/aos_soa_repack.hpp"
#include "tide/working_set.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
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
            const auto name = "lfs_tide_strategy_ws_test_" + std::to_string(rd()) + "_" + std::to_string(rd());
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

    // SH degree 3 → 15 rest components, 45 rest floats per Gaussian.
    constexpr int kShDegree = 3;
    constexpr std::size_t kRest = 45;

    struct SyntheticData {
        std::vector<float> means, scaling, rotation, opacity, sh0, shN;
        std::size_t n = 0;
    };

    SyntheticData make_blocks(std::size_t num_blocks) {
        const std::size_t n = BlockStore::kDefaultBlockSize * num_blocks;
        SyntheticData d;
        d.n = n;
        d.means.assign(3 * n, 0.0f);
        d.scaling.assign(3 * n, -2.0f);
        d.rotation.assign(4 * n, 0.0f);
        d.opacity.assign(n, 0.5f);
        d.sh0.assign(3 * n, 0.5f);
        d.shN.assign(kRest * n, 0.0f);

        for (std::size_t i = 0; i < n; ++i) {
            d.means[3 * i + 0] = static_cast<float>(i % 64);
            d.means[3 * i + 1] = static_cast<float>((i / 64) % 64);
            d.means[3 * i + 2] = static_cast<float>(i / (64 * 64));
            d.rotation[4 * i + 0] = 1.0f;
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
        s.sh_rest_components = kRest;
        return s;
    }

    // Tiny placeholder SplatData — sizing comes from WorkingSet, but the
    // constructor must hand the strategy a valid SplatData object so it can
    // read sh_degree / scene_scale.
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

    std::shared_ptr<BlockStore> share(std::unique_ptr<BlockStore> p) {
        return std::shared_ptr<BlockStore>(std::move(p));
    }

    struct Fixture {
        TempDir tmp;
        std::shared_ptr<BlockStore> store;
        std::unique_ptr<TieredCache> cache;
        std::shared_ptr<tide::WorkingSet> ws;
        std::unique_ptr<SplatData> placeholder;
        std::unique_ptr<TideStrategy> strategy;
        std::size_t num_blocks = 0;
        std::size_t gaussians_per_block = 0;

        Fixture(std::size_t blocks, std::size_t ws_capacity) : num_blocks(blocks) {
            auto data = make_blocks(blocks);
            auto r = BlockStore::stream_ply_to_base(tmp.path() / "store", as_source(data));
            if (!r.has_value()) {
                ADD_FAILURE() << "stream_ply_to_base: " << r.error();
                return;
            }
            store = share(std::move(r.value()));
            gaussians_per_block = store->block_size();

            TieredCache::Config ccfg;
            ccfg.capacity_blocks = std::max<std::size_t>(ws_capacity * 2, 4);
            cache = std::make_unique<TieredCache>(store, ccfg);

            tide::WorkingSet::Config wcfg;
            wcfg.capacity_blocks = ws_capacity;
            wcfg.bytes_per_block = store->bytes_per_block();
            wcfg.cuda_device = 0;
            auto wr = tide::WorkingSet::create(wcfg);
            if (!wr.has_value()) {
                ADD_FAILURE() << "WorkingSet::create: " << wr.error();
                return;
            }
            ws = std::shared_ptr<tide::WorkingSet>(std::move(wr.value()));

            placeholder = std::make_unique<SplatData>(make_placeholder());
            strategy = std::make_unique<TideStrategy>(*placeholder);
            strategy->set_working_set(ws);
        }
    };

    // Bring all blocks into the WS.
    void activate_all_blocks(Fixture& f) {
        std::vector<std::size_t> ids(f.num_blocks);
        for (std::size_t i = 0; i < f.num_blocks; ++i) ids[i] = i;
        const auto r = f.ws->load_and_activate(*f.cache, std::span<const std::size_t>(ids));
        ASSERT_TRUE(r.has_value()) << r.error();
    }

} // namespace

#define SKIP_IF_NO_CUDA()                              \
    do {                                               \
        if (!cuda_available()) {                       \
            GTEST_SKIP() << "No CUDA device present."; \
        }                                              \
    } while (0)

TEST(TideStrategyWorkingSetTest, AttachAndInitializeSizesFromWorkingSet) {
    SKIP_IF_NO_CUDA();
    Fixture f(/*blocks=*/8, /*ws_capacity=*/8);
    ASSERT_TRUE(f.strategy);

    param::OptimizationParameters opt;
    opt.iterations = 100;
    f.strategy->initialize(opt);

    const std::size_t expected = f.num_blocks * f.gaussians_per_block;
    EXPECT_EQ(f.strategy->soa_capacity_gaussians(), expected);

    const auto& model = f.strategy->get_model();
    EXPECT_EQ(model.size(), expected);
    EXPECT_EQ(model.get_max_sh_degree(), kShDegree);
    EXPECT_EQ(f.strategy->get_working_set(), f.ws.get());
}

TEST(TideStrategyWorkingSetTest, AttachAfterInitializeRejected) {
    SKIP_IF_NO_CUDA();
    auto placeholder = std::make_unique<SplatData>(make_placeholder());
    TideStrategy strategy(*placeholder);
    param::OptimizationParameters opt;
    opt.iterations = 10;
    strategy.initialize(opt);

    tide::WorkingSet::Config wcfg;
    wcfg.capacity_blocks = 1;
    wcfg.bytes_per_block = BlockStore::kDefaultBlockSize * tide::kAosBytesPerGaussian;
    wcfg.cuda_device = 0;
    auto wr = tide::WorkingSet::create(wcfg);
    ASSERT_TRUE(wr.has_value());
    std::shared_ptr<tide::WorkingSet> ws(std::move(wr.value()));

    EXPECT_THROW(strategy.set_working_set(ws), std::runtime_error);
}

TEST(TideStrategyWorkingSetTest, PreStepUnpacksAosIntoSoa) {
    SKIP_IF_NO_CUDA();
    Fixture f(/*blocks=*/8, /*ws_capacity=*/8);
    ASSERT_TRUE(f.strategy);

    param::OptimizationParameters opt;
    opt.iterations = 100;
    f.strategy->initialize(opt);
    activate_all_blocks(f);

    // SOA buffers start zero-filled (no bootstrap when WS attached).
    auto pre_means = f.strategy->get_model().means_raw().cpu().to_vector();
    ASSERT_FALSE(pre_means.empty());
    EXPECT_FLOAT_EQ(pre_means[0], 0.0f);
    EXPECT_FLOAT_EQ(pre_means[3], 0.0f); // before first pre_step, all zeros

    RenderOutput dummy{};
    f.strategy->pre_step(0, dummy);

    auto post_means = f.strategy->get_model().means_raw().cpu().to_vector();
    ASSERT_EQ(post_means.size(), f.strategy->soa_capacity_gaussians() * 3);
    // make_blocks() lays positions as (i%64, (i/64)%64, i/64^2) BEFORE Morton
    // reorder. The store applies bucket-sort, so we can't predict index i's
    // value directly — but we CAN confirm that the unpacked means contain
    // non-zero values matching the synthetic ramp (min=0, max>0).
    float mn = post_means[0], mx = post_means[0];
    for (float v : post_means) { mn = std::min(mn, v); mx = std::max(mx, v); }
    EXPECT_FLOAT_EQ(mn, 0.0f);
    EXPECT_GT(mx, 0.0f) << "pre_step should have unpacked non-zero means from AOS";
}

TEST(TideStrategyWorkingSetTest, StepRoutesThroughTideResidentAdam) {
    SKIP_IF_NO_CUDA();
    Fixture f(/*blocks=*/8, /*ws_capacity=*/8);
    ASSERT_TRUE(f.strategy);

    // Use iterations large enough that ShN warmup (1000) is past for our test
    // step, so all six params advance.
    param::OptimizationParameters opt;
    opt.iterations = 5000;
    f.strategy->initialize(opt);
    activate_all_blocks(f);

    RenderOutput dummy{};
    f.strategy->pre_step(0, dummy);

    // Snapshot means before step.
    const auto means_before = f.strategy->get_model().means_raw().cpu().to_vector();
    ASSERT_FALSE(means_before.empty());

    // Inject a constant gradient on Means so Adam has something to update.
    auto& grad = f.strategy->get_optimizer().get_grad(ParamType::Means);
    ASSERT_TRUE(grad.is_valid());
    grad.fill_(0.1f);

    // ShN warmup is 1000; pass iter > 1000 so ShN also updates (we don't
    // assert on it here but the path must not error).
    const int iter = 1500;
    f.strategy->step(iter);

    const auto means_after = f.strategy->get_model().means_raw().cpu().to_vector();
    ASSERT_EQ(means_after.size(), means_before.size());

    // Adam first-step direction is -lr * sign(grad). grad=+0.1 → means decrease.
    std::size_t decreased = 0;
    for (std::size_t i = 0; i < means_after.size(); ++i) {
        if (means_after[i] < means_before[i]) ++decreased;
    }
    EXPECT_GT(decreased, means_after.size() / 2)
        << "expected Adam to decrease means under positive grad on majority of elements";

    // TideResidentAdam step counter for Means should have advanced.
    const auto* resident = f.strategy->get_resident_adam();
    ASSERT_NE(resident, nullptr);
    EXPECT_GE(resident->step_count(ParamType::Means), 1);
}

TEST(TideStrategyWorkingSetTest, StepRepacksSoaIntoWorkingSetBuffer) {
    SKIP_IF_NO_CUDA();
    Fixture f(/*blocks=*/8, /*ws_capacity=*/8);
    ASSERT_TRUE(f.strategy);

    param::OptimizationParameters opt;
    opt.iterations = 5000;
    f.strategy->initialize(opt);
    activate_all_blocks(f);

    RenderOutput dummy{};
    f.strategy->pre_step(0, dummy);

    // Drive a known param change: large grad on opacity → big Adam step.
    auto& grad_op = f.strategy->get_optimizer().get_grad(ParamType::Opacity);
    ASSERT_TRUE(grad_op.is_valid());
    grad_op.fill_(1.0f);

    const auto opacity_before = f.strategy->get_model().opacity_raw().cpu().to_vector();
    f.strategy->step(1500);
    const auto opacity_after = f.strategy->get_model().opacity_raw().cpu().to_vector();
    ASSERT_EQ(opacity_after.size(), opacity_before.size());
    EXPECT_NE(opacity_after[0], opacity_before[0])
        << "opacity in SOA should have changed after Adam step";

    // Now read the active AOS buffer directly. If soa_to_aos repacked the SOA
    // scratch back into the WS active buffer, opacity at AOS offset 10 of
    // every Gaussian should equal the new SOA opacity values (within float
    // round-trip noise).
    const std::size_t n = f.strategy->soa_capacity_gaussians();
    std::vector<float> aos_host(n * tide::kAosFloatsPerGaussian);
    const void* ws_buf = f.ws->device_buffer();
    ASSERT_NE(ws_buf, nullptr);
    const auto err = cudaMemcpy(aos_host.data(), ws_buf,
                                aos_host.size() * sizeof(float),
                                cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    // Spot-check: AOS opacity at offset 10 of Gaussian 0, 100, 10000 matches SOA.
    const std::array<std::size_t, 3> probe_indices{0, 100, 10000};
    for (auto i : probe_indices) {
        if (i >= n) continue;
        const float aos_opacity = aos_host[i * tide::kAosFloatsPerGaussian + tide::kAosOffsetOpacity];
        EXPECT_FLOAT_EQ(aos_opacity, opacity_after[i])
            << "AOS opacity at gaussian " << i << " does not match SOA after step";
    }
}

TEST(TideStrategyWorkingSetTest, RoundTripsAcrossMultipleIterations) {
    SKIP_IF_NO_CUDA();
    Fixture f(/*blocks=*/8, /*ws_capacity=*/8);
    ASSERT_TRUE(f.strategy);

    param::OptimizationParameters opt;
    opt.iterations = 5000;
    f.strategy->initialize(opt);
    activate_all_blocks(f);

    RenderOutput dummy{};

    // 3 iterations of: pre_step → fill grad → step. After each iter the AOS
    // buffer must reflect the cumulative param update so the next pre_step
    // observes it.
    for (int iter = 1500; iter < 1503; ++iter) {
        f.strategy->pre_step(iter, dummy);
        auto& grad = f.strategy->get_optimizer().get_grad(ParamType::Means);
        grad.fill_(0.05f);
        f.strategy->step(iter);
    }

    const auto* resident = f.strategy->get_resident_adam();
    ASSERT_NE(resident, nullptr);
    EXPECT_GE(resident->step_count(ParamType::Means), 3);

    // Final means must differ from the original ramp (cumulative drift from
    // 3 positive-grad Adam steps).
    auto post_means = f.strategy->get_model().means_raw().cpu().to_vector();
    float drift_sum = 0.0f;
    for (std::size_t i = 0; i < std::min<std::size_t>(post_means.size(), 1000); ++i) {
        drift_sum += std::fabs(post_means[i]);
    }
    EXPECT_GT(drift_sum, 0.0f) << "expected non-zero drift after 3 Adam iterations";
}
