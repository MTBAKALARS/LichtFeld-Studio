/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file aos_soa_repack.cu
 * @brief CUDA kernels for converting between BlockStore's AOS Gaussian
 *        layout (59 contiguous floats per Gaussian) and SplatData's SOA
 *        per-attribute tensors.
 *
 * Phase 3.2a — bridges the WorkingSet (AOS, matches on-disk BlockStore
 * format) and the trainer/rasterizer (SOA tensors via SplatData).
 *
 * Performance notes:
 *   - Each thread handles one Gaussian. Loads are coalesced within a warp
 *     because consecutive threads read consecutive 59-float records.
 *   - Memory-bound: ~236 B read + 236 B written per Gaussian. On a 1 TB/s
 *     RTX 4090 HBM that's ~2 ns/Gaussian, so 1 M Gaussians ≈ 2 ms.
 *   - No shared memory, no atomics, no branching past the SH-N tail.
 */

#include "tide/aos_soa_repack.hpp"

#include <cuda_runtime.h>

namespace lfs::training::tide {

    namespace {

        constexpr int kBlockThreads = 256;

        __device__ inline std::size_t aos_offset(std::size_t gauss_idx) {
            return gauss_idx * kAosFloatsPerGaussian;
        }

        // ---- Unpack: AOS → SOA ----
        __global__ void aos_to_soa_kernel(const float* __restrict__ aos,
                                          float* __restrict__ means,
                                          float* __restrict__ scaling,
                                          float* __restrict__ rotation,
                                          float* __restrict__ opacity,
                                          float* __restrict__ sh0,
                                          float* __restrict__ shN,
                                          std::size_t num_gaussians,
                                          std::size_t shN_floats) {
            const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= num_gaussians)
                return;

            const float* g = aos + aos_offset(i);

            // means [N, 3]
            means[i * 3 + 0] = g[kAosOffsetMeans + 0];
            means[i * 3 + 1] = g[kAosOffsetMeans + 1];
            means[i * 3 + 2] = g[kAosOffsetMeans + 2];

            // scaling [N, 3]
            scaling[i * 3 + 0] = g[kAosOffsetScaling + 0];
            scaling[i * 3 + 1] = g[kAosOffsetScaling + 1];
            scaling[i * 3 + 2] = g[kAosOffsetScaling + 2];

            // rotation [N, 4]
            rotation[i * 4 + 0] = g[kAosOffsetRotation + 0];
            rotation[i * 4 + 1] = g[kAosOffsetRotation + 1];
            rotation[i * 4 + 2] = g[kAosOffsetRotation + 2];
            rotation[i * 4 + 3] = g[kAosOffsetRotation + 3];

            // opacity [N, 1]
            opacity[i] = g[kAosOffsetOpacity];

            // sh0 [N, 1, 3] = 3 floats
            sh0[i * 3 + 0] = g[kAosOffsetSh0 + 0];
            sh0[i * 3 + 1] = g[kAosOffsetSh0 + 1];
            sh0[i * 3 + 2] = g[kAosOffsetSh0 + 2];

            // shN [N, sh_rest_components, 3], flattened. Copy only what the
            // caller asked for; ignore any AOS rest floats past that point.
            if (shN != nullptr) {
                #pragma unroll
                for (std::size_t k = 0; k < kAosShNMaxFloats; ++k) {
                    if (k >= shN_floats)
                        break;
                    shN[i * shN_floats + k] = g[kAosOffsetShN + k];
                }
            }
        }

        // ---- Pack: SOA → AOS ----
        __global__ void soa_to_aos_kernel(const float* __restrict__ means,
                                          const float* __restrict__ scaling,
                                          const float* __restrict__ rotation,
                                          const float* __restrict__ opacity,
                                          const float* __restrict__ sh0,
                                          const float* __restrict__ shN,
                                          float* __restrict__ aos,
                                          std::size_t num_gaussians,
                                          std::size_t shN_floats) {
            const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= num_gaussians)
                return;

            float* g = aos + aos_offset(i);

            g[kAosOffsetMeans + 0] = means[i * 3 + 0];
            g[kAosOffsetMeans + 1] = means[i * 3 + 1];
            g[kAosOffsetMeans + 2] = means[i * 3 + 2];

            g[kAosOffsetScaling + 0] = scaling[i * 3 + 0];
            g[kAosOffsetScaling + 1] = scaling[i * 3 + 1];
            g[kAosOffsetScaling + 2] = scaling[i * 3 + 2];

            g[kAosOffsetRotation + 0] = rotation[i * 4 + 0];
            g[kAosOffsetRotation + 1] = rotation[i * 4 + 1];
            g[kAosOffsetRotation + 2] = rotation[i * 4 + 2];
            g[kAosOffsetRotation + 3] = rotation[i * 4 + 3];

            g[kAosOffsetOpacity] = opacity[i];

            g[kAosOffsetSh0 + 0] = sh0[i * 3 + 0];
            g[kAosOffsetSh0 + 1] = sh0[i * 3 + 1];
            g[kAosOffsetSh0 + 2] = sh0[i * 3 + 2];

            // shN: write the live coefficients and zero-pad the rest so the
            // BlockStore invariant (rest is always 45 floats) is preserved.
            #pragma unroll
            for (std::size_t k = 0; k < kAosShNMaxFloats; ++k) {
                if (shN != nullptr && k < shN_floats) {
                    g[kAosOffsetShN + k] = shN[i * shN_floats + k];
                } else {
                    g[kAosOffsetShN + k] = 0.0f;
                }
            }
        }

        bool views_valid_for_repack(const SoaViews& soa) {
            if (soa.num_gaussians == 0)
                return true; // No-op is always valid.
            if (soa.shN_floats_per_gaussian > kAosShNMaxFloats)
                return false;
            return soa.means_ptr != nullptr &&
                   soa.scaling_ptr != nullptr &&
                   soa.rotation_ptr != nullptr &&
                   soa.opacity_ptr != nullptr &&
                   soa.sh0_ptr != nullptr;
        }

    } // namespace

    int aos_to_soa(const float* aos_buffer,
                   const SoaViews& soa,
                   cudaStream_t stream) {
        if (soa.num_gaussians == 0)
            return 0;
        if (aos_buffer == nullptr || !views_valid_for_repack(soa))
            return static_cast<int>(cudaErrorInvalidValue);

        const std::size_t n = soa.num_gaussians;
        const dim3 threads(kBlockThreads);
        const dim3 blocks(static_cast<unsigned int>((n + kBlockThreads - 1) / kBlockThreads));

        aos_to_soa_kernel<<<blocks, threads, 0, stream>>>(
            aos_buffer,
            soa.means_ptr,
            soa.scaling_ptr,
            soa.rotation_ptr,
            soa.opacity_ptr,
            soa.sh0_ptr,
            soa.shN_ptr,
            n,
            soa.shN_floats_per_gaussian);

        return static_cast<int>(cudaGetLastError());
    }

    int soa_to_aos(const SoaViews& soa,
                   float* aos_buffer,
                   cudaStream_t stream) {
        if (soa.num_gaussians == 0)
            return 0;
        if (aos_buffer == nullptr || !views_valid_for_repack(soa))
            return static_cast<int>(cudaErrorInvalidValue);

        const std::size_t n = soa.num_gaussians;
        const dim3 threads(kBlockThreads);
        const dim3 blocks(static_cast<unsigned int>((n + kBlockThreads - 1) / kBlockThreads));

        soa_to_aos_kernel<<<blocks, threads, 0, stream>>>(
            soa.means_ptr,
            soa.scaling_ptr,
            soa.rotation_ptr,
            soa.opacity_ptr,
            soa.sh0_ptr,
            soa.shN_ptr,
            aos_buffer,
            n,
            soa.shN_floats_per_gaussian);

        return static_cast<int>(cudaGetLastError());
    }

} // namespace lfs::training::tide
