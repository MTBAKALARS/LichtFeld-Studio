/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>

// Forward declaration to keep CUDA out of the public header.
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t; // NOLINT(modernize-use-using)

namespace lfs::training::tide {

    /**
     * @brief Per-attribute pointers into Gaussian SOA scratch buffers.
     *
     * Layouts match SplatData's raw tensors:
     *   - means    : [N, 3]   (3 floats / Gaussian)
     *   - scaling  : [N, 3]   (3 floats / Gaussian, log-space)
     *   - rotation : [N, 4]   (4 floats / Gaussian, quaternion xyzw)
     *   - opacity  : [N, 1]   (1 float  / Gaussian, pre-sigmoid)
     *   - sh0      : [N, 1, 3] = [N, 3]   (DC SH coefficient, RGB)
     *   - shN      : [N, sh_rest_components, 3], flattened to
     *                shN_floats_per_gaussian floats / Gaussian.
     *                Typical values:
     *                  - SH degree 0: 0 floats
     *                  - SH degree 1: 9 floats  (3 components × 3)
     *                  - SH degree 2: 24 floats (8 components × 3)
     *                  - SH degree 3: 45 floats (15 components × 3, the max
     *                                            stored in BlockStore's `rest`)
     *
     * All pointers are device pointers. May be `nullptr` if a particular
     * attribute is unused (e.g. `shN_ptr == nullptr` when SH degree is 0).
     */
    struct SoaViews {
        float* means_ptr = nullptr;
        float* scaling_ptr = nullptr;
        float* rotation_ptr = nullptr;
        float* opacity_ptr = nullptr;
        float* sh0_ptr = nullptr;
        float* shN_ptr = nullptr;
        std::size_t num_gaussians = 0;
        std::size_t shN_floats_per_gaussian = 0; ///< Must be 0..45.
    };

    /**
     * @brief BlockStore AOS layout constants. See block_store.hpp.
     *
     * Per-Gaussian: [xyz(3) | scaling(3) | rotation(4) | opacity(1) |
     *                dc=sh0(3) | rest=shN(45)] = 59 floats = 236 bytes.
     */
    inline constexpr std::size_t kAosFloatsPerGaussian = 59;
    inline constexpr std::size_t kAosBytesPerGaussian = kAosFloatsPerGaussian * sizeof(float);
    inline constexpr std::size_t kAosOffsetMeans = 0;
    inline constexpr std::size_t kAosOffsetScaling = 3;
    inline constexpr std::size_t kAosOffsetRotation = 6;
    inline constexpr std::size_t kAosOffsetOpacity = 10;
    inline constexpr std::size_t kAosOffsetSh0 = 11;
    inline constexpr std::size_t kAosOffsetShN = 14;
    inline constexpr std::size_t kAosShNMaxFloats = 45;

    /**
     * @brief Unpack the WorkingSet's AOS-interleaved device buffer into per-
     *        attribute SOA scratch buffers.
     *
     * Each Gaussian's 59 contiguous floats are scattered into the six SOA
     * tensors at the same global index. Operates on `soa.num_gaussians`
     * Gaussians starting at the front of `aos_buffer`.
     *
     * For SH degrees < 3, only the first `soa.shN_floats_per_gaussian`
     * floats of the AOS `rest` slot are written to `shN_ptr`; the higher
     * AOS coefficients are ignored. If `soa.shN_ptr == nullptr` the shN
     * attribute is skipped entirely.
     *
     * Thread-safe with respect to other kernels on different streams.
     * Returns 0 on success, non-zero `cudaError_t` cast to int on launch
     * failure.
     *
     * @param aos_buffer Device pointer to the AOS buffer. Must contain at
     *                   least `soa.num_gaussians * kAosFloatsPerGaussian`
     *                   floats of valid data.
     * @param soa        Output SOA views (device pointers).
     * @param stream     CUDA stream (nullptr = default stream).
     */
    int aos_to_soa(const float* aos_buffer,
                   const SoaViews& soa,
                   cudaStream_t stream);

    /**
     * @brief Pack per-attribute SOA tensors back into the WorkingSet's
     *        AOS-interleaved device buffer.
     *
     * Inverse of @ref aos_to_soa. For SH degrees < 3, the unused trailing
     * floats of the AOS `rest` slot are zero-filled to preserve the
     * BlockStore on-disk invariant (rest is always 45 floats, padded with
     * zeros for lower SH degrees). If `soa.shN_ptr == nullptr` the full
     * 45-float rest region is zero-filled.
     *
     * Returns 0 on success, non-zero `cudaError_t` cast to int on launch
     * failure.
     */
    int soa_to_aos(const SoaViews& soa,
                   float* aos_buffer,
                   cudaStream_t stream);

} // namespace lfs::training::tide
