/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file frustum_culler.cpp
 * @brief V1 CPU sphere-vs-6-plane culling. ~3M FMAs per 1B-Gaussian frame; <1 ms.
 */

#include "tide/frustum_culler.hpp"

#include "core/camera.hpp"

#include <cmath>
#include <stdexcept>

namespace lfs::training::tide {

    bool FrustumCuller::visible(const lfs::core::BlockStore::BlockBounds& b,
                                std::span<const Plane, 6> planes) noexcept {
        for (std::size_t i = 0; i < 6; ++i) {
            const Plane& p = planes[i];
            const float dist = p.a * b.center[0] + p.b * b.center[1] +
                               p.c * b.center[2] + p.d;
            if (dist < -b.radius) return false;
        }
        return true;
    }

    void FrustumCuller::cull(std::span<const lfs::core::BlockStore::BlockBounds> bounds,
                             std::span<const Plane, 6> planes,
                             std::vector<std::size_t>& out) {
        out.clear();
        out.reserve(bounds.size() / 4); // optimistic
        for (std::size_t id = 0; id < bounds.size(); ++id) {
            if (visible(bounds[id], planes)) {
                out.push_back(id);
            }
        }
        // Already ascending because we iterate in id order.
    }

    void FrustumCuller::compute_frustum_planes(const lfs::core::Camera& cam,
                                               float near_plane,
                                               float far_plane,
                                               std::array<Plane, 6>& out) {
        if (!(near_plane > 0.0f) || !(far_plane > near_plane)) {
            throw std::runtime_error(
                "FrustumCuller::compute_frustum_planes: require 0 < near < far");
        }

        // Pull the 4x4 world->camera matrix down to CPU. world_view_transform
        // is shape [1, 4, 4] (per camera.cpp). We need the 3x3 rotation R and
        // 3-vector translation t such that p_camera = R * p_world + t.
        const auto& w2c_tensor = cam.world_view_transform();
        auto w2c_cpu = w2c_tensor.squeeze(0).cpu();
        auto w2c = w2c_cpu.accessor<float, 2>();

        const float R[3][3] = {
            {w2c(0, 0), w2c(0, 1), w2c(0, 2)},
            {w2c(1, 0), w2c(1, 1), w2c(1, 2)},
            {w2c(2, 0), w2c(2, 1), w2c(2, 2)},
        };
        const float t[3] = {w2c(0, 3), w2c(1, 3), w2c(2, 3)};

        // Camera-space half-tangents from intrinsics. COLMAP/3DGS convention:
        // camera looks down +Z. A point (xc,yc,zc) projects inside the image
        // iff |xc| <= zc * tan(fovx/2) and |yc| <= zc * tan(fovy/2).
        // We use the precomputed FoVx/FoVy stored on the Camera, with a
        // small fallback to intrinsics if those are zero (defensive — every
        // Camera in the codebase populates FoV in its constructor).
        float fovx = cam.FoVx();
        float fovy = cam.FoVy();
        if (fovx <= 0.0f) {
            fovx = 2.0f * std::atan(static_cast<float>(cam.camera_width()) /
                                    (2.0f * cam.focal_x()));
        }
        if (fovy <= 0.0f) {
            fovy = 2.0f * std::atan(static_cast<float>(cam.camera_height()) /
                                    (2.0f * cam.focal_y()));
        }
        const float tx = std::tan(fovx * 0.5f);
        const float ty = std::tan(fovy * 0.5f);

        // Camera-space planes (a,b,c,d) with `a*x + b*y + c*z + d >= 0`
        // meaning "on the visible side". Origin (0,0,0) is the camera itself;
        // visible volume is {zc > near, zc < far, |xc| < zc*tx, |yc| < zc*ty}.
        //
        //   left:   xc + tx*zc >= 0
        //   right:  -xc + tx*zc >= 0
        //   bottom: yc + ty*zc >= 0
        //   top:    -yc + ty*zc >= 0
        //   near:   zc - near >= 0
        //   far:    -zc + far >= 0
        const Plane cam_planes[6] = {
            { 1.0f,  0.0f,   tx,  0.0f},        // left
            {-1.0f,  0.0f,   tx,  0.0f},        // right
            { 0.0f,  1.0f,   ty,  0.0f},        // bottom
            { 0.0f, -1.0f,   ty,  0.0f},        // top
            { 0.0f,  0.0f,  1.0f, -near_plane}, // near
            { 0.0f,  0.0f, -1.0f,  far_plane},  // far
        };

        // Transform each plane from camera space to world space.
        // A point p_w satisfies the world plane iff p_c = R*p_w + t satisfies
        // the camera plane. Substituting:
        //   n_c . (R*p_w + t) + d_c = (R^T * n_c) . p_w + (n_c . t + d_c) = 0
        // So n_w = R^T * n_c and d_w = n_c . t + d_c. R is row-major; with
        // R[i][j] = row i col j, R^T has element (k,i) = R[i][k], hence
        // n_w[k] = sum_i R[i][k] * n_c[i].
        for (std::size_t k = 0; k < 6; ++k) {
            const Plane& pc = cam_planes[k];
            float n_w[3];
            for (std::size_t j = 0; j < 3; ++j) {
                n_w[j] = R[0][j] * pc.a + R[1][j] * pc.b + R[2][j] * pc.c;
            }
            const float d_w = pc.a * t[0] + pc.b * t[1] + pc.c * t[2] + pc.d;

            // Normalize so |n| = 1 → BlockBounds::radius can be compared
            // directly against the signed distance.
            const float inv_len = 1.0f / std::sqrt(n_w[0] * n_w[0] +
                                                   n_w[1] * n_w[1] +
                                                   n_w[2] * n_w[2]);
            out[k] = Plane{n_w[0] * inv_len, n_w[1] * inv_len, n_w[2] * inv_len,
                           d_w * inv_len};
        }
    }

} // namespace lfs::training::tide
