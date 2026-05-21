/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file frustum_culler.cpp
 * @brief V1 CPU sphere-vs-6-plane culling. ~3M FMAs per 1B-Gaussian frame; <1 ms.
 */

#include "tide/frustum_culler.hpp"

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

} // namespace lfs::training::tide
