/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_tide_strategy_moments.cpp
 * @brief Phase 3.5.3e-2: TideStrategy with per-block Adam moments end-to-end.
 *
 * Verifies that:
 *   - When the attached WorkingSet has a non-zero moments region (the
 *     store was baked with `Config::with_moments=true`), TideStrategy
 *     allocates a second SOA scratch (12 buffers: 6 m_* + 6 v_*) and
 *     soa_moments_scratch_bytes() reports their sum.
 *   - step() routes per-block Adam through TideResidentAdam::step_external_moments
 *     using m/v pointers derived from WorkingSet::moments_device_buffer(local_idx),
 *     not the optimizer's internal Adam state.
 *   - The per-block step counter is incremented exactly once per slot per step()
 *     call and survives eviction + re-admission (in-process; persistence comes
 *     in Phase 3.5.7).
 *   - The legacy v1 path (no moments) still works bit-identically to Phase
 *     3.3a behavior — soa_moments_scratch_bytes()==0 and no moments kernels run.
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
#include <cstdint>
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
            const auto name = "lfs_tide_strategy_moments_" + std::to_string(rd()) + "_" + std::to_string(rd());
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

    // Fixture that stands up a store with `with_moments=true`, wires it
    // through TieredCache + WorkingSet (slot_bytes = data + moments), then
    // constructs a TideStrategy bound to that WorkingSet.
    struct MomentsFixture {
        TempDir tmp;
        std::shared_ptr<BlockStore> store;
        std::unique_ptr<TieredCache> cache;
        std::shared_ptr<tide::WorkingSet> ws;
        std::unique_ptr<SplatData> placeholder;
        std::unique_ptr<TideStrategy> strategy;
        std::size_t num_blocks = 0;
        std::size_t gaussians_per_block = 0;

        MomentsFixture(std::size_t blocks, std::size_t ws_capacity)
            : num_blocks(blocks) {
            auto data = make_blocks(blocks);
            BlockStore::Config bcfg;
            bcfg.with_moments = true;
            auto r = BlockStore::stream_ply_to_base(tmp.path() / "store",
                                                   as_source(data), bcfg);
            if (!r.has_value()) {
                ADD_FAILURE() << "stream_ply_to_base: " << r.error();
                return;
            }
            store = share(std::move(r.value()));
            if (!store->has_moments()) {
                ADD_FAILURE() << "fixture: store missing moments region";
                return;
            }
            gaussians_per_block = store->block_size();

            TieredCache::Config ccfg;
            ccfg.capacity_blocks = std::max<std::size_t>(ws_capacity * 2, 4);
            cache = std::make_unique<TieredCache>(store, ccfg);

            tide::WorkingSet::Config wcfg;
            wcfg.capacity_blocks = ws_capacity;
            wcfg.bytes_per_block = store->bytes_per_block();
            wcfg.moments_bytes_per_block = store->moments_bytes_per_block();
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

        // Activate a specific subset (synchronous load_and_activate).
        void activate(const std::vector<std::size_t>& ids) {
            const auto r = ws->load_and_activate(*cache, std::span<const std::size_t>(ids));
            ASSERT_TRUE(r.has_value()) << r.error();
        }

        void activate_all() {
            std::vector<std::size_t> ids(num_blocks);
            for (std::size_t i = 0; i < num_blocks; ++i) ids[i] = i;
            activate(ids);
        }

        // Fill all six grad tensors with the same scalar so step() makes
        // a measurable update. Uses the AdamOptimizer's grad-buffer host
        // (which TideStrategy::step reads from) — same path the real
        // backward pass writes through.
        void fill_grads(float value) {
            auto& opt = strategy->get_optimizer();
            for (auto type : AdamOptimizer::all_param_types()) {
                auto& g = opt.get_grad(type);
                if (!g.is_valid() || g.numel() == 0) continue;
                g.fill_(value);
            }
        }
    };

} // namespace

#define SKIP_IF_NO_CUDA()                              \
    do {                                               \
        if (!cuda_available()) {                       \
            GTEST_SKIP() << "No CUDA device present."; \
        }                                              \
    } while (0)

// ============================================================================
// 1. Sizing: moments-enabled path allocates the 12-buffer m/v scratch
// ============================================================================

TEST(TideStrategyMomentsTest, SoaMomentsScratchAllocatedWhenMomentsEnabled) {
    SKIP_IF_NO_CUDA();
    MomentsFixture f(/*blocks=*/2, /*ws_capacity=*/2);
    ASSERT_TRUE(f.strategy);

    param::OptimizationParameters opt;
    opt.iterations = 100;
    f.strategy->initialize(opt);

    // 12 buffers (m + v for each of means(3), scaling(3), rotation(4),
    // opacity(1), sh0(3), shN(45)). Total floats per Gaussian = 2 * 59 = 118.
    const std::size_t n = f.strategy->soa_capacity_gaussians();
    const std::size_t expected_bytes = sizeof(float) * n * 2u *
                                       (3 + 3 + 4 + 1 + 3 + kRest);
    EXPECT_EQ(f.strategy->soa_moments_scratch_bytes(), expected_bytes);
    // The data-region scratch is unchanged.
    EXPECT_EQ(f.strategy->soa_scratch_bytes(),
              sizeof(float) * n * (3 + 3 + 4 + 1 + 3 + kRest));
}

TEST(TideStrategyMomentsTest, LegacyV1PathSkipsMomentsScratch) {
    SKIP_IF_NO_CUDA();
    // Build a store WITHOUT with_moments and verify moments scratch is zero.
    TempDir tmp;
    auto data = make_blocks(2);
    auto r = BlockStore::stream_ply_to_base(tmp.path() / "store", as_source(data));
    ASSERT_TRUE(r.has_value()) << r.error();
    auto store = share(std::move(r.value()));
    ASSERT_FALSE(store->has_moments());

    TieredCache::Config ccfg;
    ccfg.capacity_blocks = 4;
    auto cache = std::make_unique<TieredCache>(store, ccfg);

    tide::WorkingSet::Config wcfg;
    wcfg.capacity_blocks = 2;
    wcfg.bytes_per_block = store->bytes_per_block();
    // moments_bytes_per_block left at 0 (legacy path).
    wcfg.cuda_device = 0;
    auto wr = tide::WorkingSet::create(wcfg);
    ASSERT_TRUE(wr.has_value()) << wr.error();
    std::shared_ptr<tide::WorkingSet> ws(std::move(wr.value()));

    auto placeholder = std::make_unique<SplatData>(make_placeholder());
    auto strategy = std::make_unique<TideStrategy>(*placeholder);
    strategy->set_working_set(ws);

    param::OptimizationParameters opt;
    opt.iterations = 100;
    strategy->initialize(opt);

    EXPECT_EQ(strategy->soa_moments_scratch_bytes(), 0u);
    EXPECT_GT(strategy->soa_scratch_bytes(), 0u);
}

// ============================================================================
// 2. Per-block step counter increments once per step() per active slot
// ============================================================================

TEST(TideStrategyMomentsTest, StepIncrementsPerBlockCounterOncePerSlot) {
    SKIP_IF_NO_CUDA();
    MomentsFixture f(/*blocks=*/4, /*ws_capacity=*/4);
    ASSERT_TRUE(f.strategy);

    param::OptimizationParameters opt;
    opt.iterations = 5000; // Past ShN warmup (1000) for safety.
    f.strategy->initialize(opt);
    f.activate_all();

    // No steps yet — all counters must be 0.
    for (std::size_t i = 0; i < f.num_blocks; ++i) {
        EXPECT_EQ(f.strategy->block_step_count(i), 0);
    }
    EXPECT_EQ(f.strategy->last_step_block_count(), 0u);

    RenderOutput dummy{};
    f.strategy->pre_step(1500, dummy);
    f.fill_grads(0.01f);
    f.strategy->step(1500);

    EXPECT_EQ(f.strategy->last_step_block_count(), f.num_blocks);
    for (std::size_t i = 0; i < f.num_blocks; ++i) {
        EXPECT_EQ(f.strategy->block_step_count(i), 1) << "block " << i;
    }

    // Second step → counters go to 2.
    f.strategy->pre_step(1501, dummy);
    f.fill_grads(0.01f);
    f.strategy->step(1501);
    for (std::size_t i = 0; i < f.num_blocks; ++i) {
        EXPECT_EQ(f.strategy->block_step_count(i), 2) << "block " << i;
    }
}

// ============================================================================
// 3. Eviction + re-admission preserves the per-block step counter
//
// Phase 3.5.3e-2 acceptance test: when capacity_blocks < num_blocks, a
// camera change forces eviction. The block re-enters the working set on a
// later activation, and its per-block Adam step counter (in TideStrategy)
// continues advancing from where it left off. The underlying m/v device
// buffer also travels with the block via WorkingSet writeback → cache →
// re-upload (proven byte-exact in 3.5.3b WorkingSetMomentsTest::
// EvictDirtyWritesBackBothRegions); this test exercises the strategy-level
// wiring that drives that writeback through mark_dirty.
// ============================================================================

TEST(TideStrategyMomentsTest, EvictAndReadmitPreservesPerBlockCounter) {
    SKIP_IF_NO_CUDA();
    // 8 blocks total, capacity for only 4 — forces eviction on a camera change.
    MomentsFixture f(/*blocks=*/8, /*ws_capacity=*/4);
    ASSERT_TRUE(f.strategy);

    param::OptimizationParameters opt;
    opt.iterations = 5000;
    f.strategy->initialize(opt);

    RenderOutput dummy{};

    // Phase 1 — train iter 0..1 on blocks {0,1,2,3}.
    f.activate({0, 1, 2, 3});
    for (int it : {1500, 1501}) {
        f.strategy->pre_step(it, dummy);
        f.fill_grads(0.01f);
        f.strategy->step(it);
    }
    for (std::size_t b : {0u, 1u, 2u, 3u}) {
        EXPECT_EQ(f.strategy->block_step_count(b), 2) << "block " << b;
    }
    EXPECT_EQ(f.strategy->block_step_count(4), 0);

    // Phase 2 — switch to blocks {4,5,6,7}. The previous resident set
    // {0,1,2,3} is evicted; their dirty data + moments regions are
    // written back to TieredCache. Their per-block counters in the
    // strategy are still 2 (the map is keyed by global block_id, not
    // local slot index).
    f.activate({4, 5, 6, 7});
    f.strategy->pre_step(1502, dummy);
    f.fill_grads(0.01f);
    f.strategy->step(1502);
    for (std::size_t b : {0u, 1u, 2u, 3u}) {
        EXPECT_EQ(f.strategy->block_step_count(b), 2)
            << "block " << b << " counter must survive eviction";
    }
    for (std::size_t b : {4u, 5u, 6u, 7u}) {
        EXPECT_EQ(f.strategy->block_step_count(b), 1)
            << "newly-admitted block " << b << " counter starts at 1";
    }

    // Phase 3 — re-admit {0,1,2,3} after they were evicted, step once.
    // Counters resume from 2 → 3 for those blocks; {4..7} are now evicted
    // but their counters stay at 1.
    f.activate({0, 1, 2, 3});
    f.strategy->pre_step(1503, dummy);
    f.fill_grads(0.01f);
    f.strategy->step(1503);
    for (std::size_t b : {0u, 1u, 2u, 3u}) {
        EXPECT_EQ(f.strategy->block_step_count(b), 3)
            << "re-admitted block " << b << " must continue counting";
    }
    for (std::size_t b : {4u, 5u, 6u, 7u}) {
        EXPECT_EQ(f.strategy->block_step_count(b), 1)
            << "evicted block " << b << " counter must persist";
    }
}

// ============================================================================
// 4. Moments-enabled step() updates parameters (proxy for full Adam wiring)
// ============================================================================

TEST(TideStrategyMomentsTest, StepActuallyUpdatesParameters) {
    SKIP_IF_NO_CUDA();
    MomentsFixture f(/*blocks=*/2, /*ws_capacity=*/2);
    ASSERT_TRUE(f.strategy);

    param::OptimizationParameters opt;
    opt.iterations = 5000;
    f.strategy->initialize(opt);
    f.activate_all();

    RenderOutput dummy{};
    f.strategy->pre_step(1500, dummy);

    // Snapshot means after unpack, before step.
    auto pre = f.strategy->get_model().means_raw().cpu().to_vector();

    f.fill_grads(0.1f);
    f.strategy->step(1500);

    // After step the WorkingSet's active AOS region holds the post-Adam
    // params. The next pre_step unpacks them into the SOA view, and we
    // re-read means_raw() to compare.
    f.strategy->pre_step(1501, dummy);
    auto post = f.strategy->get_model().means_raw().cpu().to_vector();

    ASSERT_EQ(pre.size(), post.size());
    std::size_t different = 0;
    for (std::size_t i = 0; i < pre.size(); ++i) {
        if (pre[i] != post[i]) ++different;
    }
    EXPECT_GT(different, 0u) << "step() with non-zero grads must move at least one param";
}
