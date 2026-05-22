/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_aos_soa_repack.cpp
 * @brief Unit tests for Phase 3.2a — AOS↔SOA repack kernels that bridge
 *        WorkingSet's BlockStore-format buffer and SplatData's SOA tensors.
 */

#include "tide/aos_soa_repack.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

    using lfs::training::tide::kAosBytesPerGaussian;
    using lfs::training::tide::kAosFloatsPerGaussian;
    using lfs::training::tide::kAosOffsetMeans;
    using lfs::training::tide::kAosOffsetOpacity;
    using lfs::training::tide::kAosOffsetRotation;
    using lfs::training::tide::kAosOffsetScaling;
    using lfs::training::tide::kAosOffsetSh0;
    using lfs::training::tide::kAosOffsetShN;
    using lfs::training::tide::kAosShNMaxFloats;
    using lfs::training::tide::SoaViews;

    // ---- Device buffer helper ----
    struct DevBuf {
        float* ptr = nullptr;
        std::size_t n = 0;

        explicit DevBuf(std::size_t num_floats) : n(num_floats) {
            if (n == 0)
                return;
            const auto err = cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(float));
            if (err != cudaSuccess) {
                ptr = nullptr;
            }
        }
        ~DevBuf() {
            if (ptr != nullptr) {
                cudaFree(ptr);
                ptr = nullptr;
            }
        }
        DevBuf(const DevBuf&) = delete;
        DevBuf& operator=(const DevBuf&) = delete;
        DevBuf(DevBuf&& o) noexcept : ptr(o.ptr), n(o.n) {
            o.ptr = nullptr;
            o.n = 0;
        }
        DevBuf& operator=(DevBuf&&) = delete;

        void upload(const std::vector<float>& src) {
            ASSERT_NE(ptr, nullptr);
            ASSERT_EQ(src.size(), n);
            ASSERT_EQ(cudaMemcpy(ptr, src.data(), n * sizeof(float),
                                 cudaMemcpyHostToDevice),
                      cudaSuccess);
        }
        std::vector<float> download() const {
            std::vector<float> out(n, 0.0f);
            if (n == 0)
                return out;
            EXPECT_EQ(cudaMemcpy(out.data(), ptr, n * sizeof(float),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            return out;
        }
        void zero() {
            if (n == 0)
                return;
            ASSERT_EQ(cudaMemset(ptr, 0, n * sizeof(float)), cudaSuccess);
        }
    };

    std::vector<float> make_aos_pattern(std::size_t num_gaussians, std::uint32_t seed) {
        std::vector<float> v(num_gaussians * kAosFloatsPerGaussian);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (auto& f : v)
            f = dist(rng);
        return v;
    }

    // Build SoaViews around freshly-allocated DevBufs for `num_gaussians`
    // with the requested shN width.
    struct SoaBundle {
        DevBuf means;
        DevBuf scaling;
        DevBuf rotation;
        DevBuf opacity;
        DevBuf sh0;
        DevBuf shN;
        SoaViews views{};

        SoaBundle(std::size_t n, std::size_t shN_floats)
            : means(n * 3),
              scaling(n * 3),
              rotation(n * 4),
              opacity(n * 1),
              sh0(n * 3),
              shN(n * shN_floats) {
            views.means_ptr = means.ptr;
            views.scaling_ptr = scaling.ptr;
            views.rotation_ptr = rotation.ptr;
            views.opacity_ptr = opacity.ptr;
            views.sh0_ptr = sh0.ptr;
            views.shN_ptr = (shN_floats == 0) ? nullptr : shN.ptr;
            views.num_gaussians = n;
            views.shN_floats_per_gaussian = shN_floats;
        }
    };

    class AosSoaRepackTest : public ::testing::Test {
    protected:
        void SetUp() override {
            if (cudaSetDevice(0) != cudaSuccess) {
                GTEST_SKIP() << "No CUDA device available.";
            }
        }
    };

    // ---------------------------------------------------------------------

    TEST_F(AosSoaRepackTest, AosToSoaScattersAllAttributesCorrectly) {
        constexpr std::size_t kN = 1024;
        constexpr std::size_t kShN = 45;

        DevBuf aos(kN * kAosFloatsPerGaussian);
        const auto host_aos = make_aos_pattern(kN, 0xA05Au);
        aos.upload(host_aos);

        SoaBundle soa(kN, kShN);
        ASSERT_EQ(lfs::training::tide::aos_to_soa(aos.ptr, soa.views, nullptr), 0);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto h_means = soa.means.download();
        const auto h_scaling = soa.scaling.download();
        const auto h_rotation = soa.rotation.download();
        const auto h_opacity = soa.opacity.download();
        const auto h_sh0 = soa.sh0.download();
        const auto h_shN = soa.shN.download();

        for (std::size_t i = 0; i < kN; ++i) {
            const float* g = &host_aos[i * kAosFloatsPerGaussian];
            for (std::size_t k = 0; k < 3; ++k) {
                EXPECT_FLOAT_EQ(h_means[i * 3 + k], g[kAosOffsetMeans + k]) << "means @ i=" << i << " k=" << k;
                EXPECT_FLOAT_EQ(h_scaling[i * 3 + k], g[kAosOffsetScaling + k]) << "scaling @ i=" << i << " k=" << k;
                EXPECT_FLOAT_EQ(h_sh0[i * 3 + k], g[kAosOffsetSh0 + k]) << "sh0 @ i=" << i << " k=" << k;
            }
            for (std::size_t k = 0; k < 4; ++k) {
                EXPECT_FLOAT_EQ(h_rotation[i * 4 + k], g[kAosOffsetRotation + k]) << "rotation @ i=" << i << " k=" << k;
            }
            EXPECT_FLOAT_EQ(h_opacity[i], g[kAosOffsetOpacity]) << "opacity @ i=" << i;
            for (std::size_t k = 0; k < kShN; ++k) {
                EXPECT_FLOAT_EQ(h_shN[i * kShN + k], g[kAosOffsetShN + k]) << "shN @ i=" << i << " k=" << k;
            }
        }
    }

    // ---------------------------------------------------------------------

    TEST_F(AosSoaRepackTest, RoundTripIsBitwiseIdentityForSh3) {
        constexpr std::size_t kN = 2048;
        constexpr std::size_t kShN = 45;

        DevBuf aos_src(kN * kAosFloatsPerGaussian);
        DevBuf aos_dst(kN * kAosFloatsPerGaussian);
        const auto host_aos = make_aos_pattern(kN, 0xBEEF0001u);
        aos_src.upload(host_aos);
        aos_dst.zero();

        SoaBundle soa(kN, kShN);
        ASSERT_EQ(lfs::training::tide::aos_to_soa(aos_src.ptr, soa.views, nullptr), 0);
        ASSERT_EQ(lfs::training::tide::soa_to_aos(soa.views, aos_dst.ptr, nullptr), 0);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto h_dst = aos_dst.download();
        ASSERT_EQ(h_dst.size(), host_aos.size());
        for (std::size_t i = 0; i < host_aos.size(); ++i) {
            EXPECT_FLOAT_EQ(h_dst[i], host_aos[i]) << "round-trip mismatch @ idx=" << i;
        }
    }

    // ---------------------------------------------------------------------

    TEST_F(AosSoaRepackTest, LowerShDegreesZeroPadAosTail) {
        // SH degree 1 → shN_floats = 9. Round-trip should preserve the first
        // 9 rest floats, and zero out the remaining 36 in AOS.
        constexpr std::size_t kN = 64;
        constexpr std::size_t kShN = 9;

        DevBuf aos_src(kN * kAosFloatsPerGaussian);
        DevBuf aos_dst(kN * kAosFloatsPerGaussian);
        auto host_aos = make_aos_pattern(kN, 0x12345678u);
        // Fill rest tail with distinctive non-zero values so we can detect
        // whether soa_to_aos correctly clobbered them with zero.
        for (std::size_t i = 0; i < kN; ++i) {
            for (std::size_t k = 0; k < kAosShNMaxFloats; ++k) {
                host_aos[i * kAosFloatsPerGaussian + kAosOffsetShN + k] = 100.0f + static_cast<float>(k);
            }
        }
        aos_src.upload(host_aos);
        aos_dst.zero();

        SoaBundle soa(kN, kShN);
        ASSERT_EQ(lfs::training::tide::aos_to_soa(aos_src.ptr, soa.views, nullptr), 0);
        ASSERT_EQ(lfs::training::tide::soa_to_aos(soa.views, aos_dst.ptr, nullptr), 0);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto h_shN = soa.shN.download();
        for (std::size_t i = 0; i < kN; ++i) {
            for (std::size_t k = 0; k < kShN; ++k) {
                EXPECT_FLOAT_EQ(h_shN[i * kShN + k], 100.0f + static_cast<float>(k))
                    << "SoA shN @ i=" << i << " k=" << k;
            }
        }

        const auto h_dst = aos_dst.download();
        for (std::size_t i = 0; i < kN; ++i) {
            // First kShN AOS rest floats should be preserved.
            for (std::size_t k = 0; k < kShN; ++k) {
                EXPECT_FLOAT_EQ(h_dst[i * kAosFloatsPerGaussian + kAosOffsetShN + k],
                                100.0f + static_cast<float>(k))
                    << "AOS rest preserved @ i=" << i << " k=" << k;
            }
            // Tail floats must be zeroed.
            for (std::size_t k = kShN; k < kAosShNMaxFloats; ++k) {
                EXPECT_FLOAT_EQ(h_dst[i * kAosFloatsPerGaussian + kAosOffsetShN + k], 0.0f)
                    << "AOS rest tail zeroed @ i=" << i << " k=" << k;
            }
            // Non-shN attributes should still round-trip exactly.
            for (std::size_t k = 0; k < kAosOffsetShN; ++k) {
                EXPECT_FLOAT_EQ(h_dst[i * kAosFloatsPerGaussian + k],
                                host_aos[i * kAosFloatsPerGaussian + k])
                    << "non-shN attr round-trip @ i=" << i << " k=" << k;
            }
        }
    }

    // ---------------------------------------------------------------------

    TEST_F(AosSoaRepackTest, ShDegreeZeroSkipsShNAndZerosAosTail) {
        // SH degree 0 → shN_floats = 0, shN_ptr = nullptr.
        constexpr std::size_t kN = 128;

        DevBuf aos_src(kN * kAosFloatsPerGaussian);
        DevBuf aos_dst(kN * kAosFloatsPerGaussian);
        auto host_aos = make_aos_pattern(kN, 0xCAFE0BABu);
        // Distinct non-zero tail to verify pack zeros it.
        for (std::size_t i = 0; i < kN; ++i) {
            for (std::size_t k = 0; k < kAosShNMaxFloats; ++k) {
                host_aos[i * kAosFloatsPerGaussian + kAosOffsetShN + k] = -7.0f - static_cast<float>(k);
            }
        }
        aos_src.upload(host_aos);
        aos_dst.zero();

        SoaBundle soa(kN, 0);
        ASSERT_EQ(soa.views.shN_ptr, nullptr);
        ASSERT_EQ(lfs::training::tide::aos_to_soa(aos_src.ptr, soa.views, nullptr), 0);
        ASSERT_EQ(lfs::training::tide::soa_to_aos(soa.views, aos_dst.ptr, nullptr), 0);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        const auto h_dst = aos_dst.download();
        for (std::size_t i = 0; i < kN; ++i) {
            // Non-shN attrs round-trip.
            for (std::size_t k = 0; k < kAosOffsetShN; ++k) {
                EXPECT_FLOAT_EQ(h_dst[i * kAosFloatsPerGaussian + k],
                                host_aos[i * kAosFloatsPerGaussian + k])
                    << "non-shN attr round-trip @ i=" << i << " k=" << k;
            }
            // Entire rest region must be zero.
            for (std::size_t k = 0; k < kAosShNMaxFloats; ++k) {
                EXPECT_FLOAT_EQ(h_dst[i * kAosFloatsPerGaussian + kAosOffsetShN + k], 0.0f)
                    << "AOS rest zeroed @ i=" << i << " k=" << k;
            }
        }
    }

    // ---------------------------------------------------------------------

    TEST_F(AosSoaRepackTest, ZeroGaussiansIsNoop) {
        SoaBundle soa(0, 0);
        EXPECT_EQ(lfs::training::tide::aos_to_soa(nullptr, soa.views, nullptr), 0);
        EXPECT_EQ(lfs::training::tide::soa_to_aos(soa.views, nullptr, nullptr), 0);
    }

    // ---------------------------------------------------------------------

    TEST_F(AosSoaRepackTest, NullPointerWithNonZeroNIsRejected) {
        SoaBundle soa(16, 45);
        // Valid views but nullptr AOS buffer.
        EXPECT_NE(lfs::training::tide::aos_to_soa(nullptr, soa.views, nullptr), 0);
        EXPECT_NE(lfs::training::tide::soa_to_aos(soa.views, nullptr, nullptr), 0);

        // Valid AOS but corrupted views (missing means_ptr).
        DevBuf aos(16 * kAosFloatsPerGaussian);
        aos.zero();
        SoaViews bad = soa.views;
        bad.means_ptr = nullptr;
        EXPECT_NE(lfs::training::tide::aos_to_soa(aos.ptr, bad, nullptr), 0);
        EXPECT_NE(lfs::training::tide::soa_to_aos(bad, aos.ptr, nullptr), 0);

        // Out-of-range shN width.
        SoaViews too_wide = soa.views;
        too_wide.shN_floats_per_gaussian = kAosShNMaxFloats + 1;
        EXPECT_NE(lfs::training::tide::aos_to_soa(aos.ptr, too_wide, nullptr), 0);
        EXPECT_NE(lfs::training::tide::soa_to_aos(too_wide, aos.ptr, nullptr), 0);
    }

    // ---------------------------------------------------------------------

    TEST_F(AosSoaRepackTest, RunsOnCustomStream) {
        constexpr std::size_t kN = 256;
        constexpr std::size_t kShN = 45;

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

        DevBuf aos(kN * kAosFloatsPerGaussian);
        const auto host_aos = make_aos_pattern(kN, 0x5723u);
        aos.upload(host_aos);

        SoaBundle soa(kN, kShN);
        EXPECT_EQ(lfs::training::tide::aos_to_soa(aos.ptr, soa.views, stream), 0);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        const auto h_means = soa.means.download();
        for (std::size_t i = 0; i < kN; ++i) {
            for (std::size_t k = 0; k < 3; ++k) {
                EXPECT_FLOAT_EQ(h_means[i * 3 + k],
                                host_aos[i * kAosFloatsPerGaussian + kAosOffsetMeans + k]);
            }
        }
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

} // namespace
