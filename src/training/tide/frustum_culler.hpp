/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/block_store.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace lfs::training::tide {

    /**
     * @brief CPU-side frustum culler over block bounding spheres.
     *
     * TideGS uses bounding spheres (not AABBs) for block-level culling
     * (arXiv:2605.20150 Sec. 3.3): a sphere is rejected if its signed distance
     * to any frustum plane is less than -radius.
     *
     * V1 implementation is single-threaded CPU. With ~250K blocks (1B Gaussians)
     * and 6-plane test = ~12 fmas per block per camera = ~3M fmas per frame.
     * That's <1 ms on a modern CPU and is not on the critical path; a CUDA port
     * is straightforward when needed.
     *
     * Plane convention: `(a, b, c, d)` where `a*x + b*y + c*z + d >= 0` means
     * the point is on the visible side. The 6 planes are passed in any order
     * (left, right, bottom, top, near, far is the conventional ordering).
     */
    class FrustumCuller {
    public:
        struct Plane {
            float a, b, c, d;
        };

        /**
         * @brief Test all block bounds against the 6 planes; return visible block_ids in order.
         *
         * @param bounds  Per-block bounding spheres, indexed by block_id.
         * @param planes  Exactly 6 frustum planes (normalized: a²+b²+c²=1).
         * @param out     Visible block_ids, sorted ascending. Cleared before fill.
         */
        static void cull(std::span<const lfs::core::BlockStore::BlockBounds> bounds,
                         std::span<const Plane, 6> planes,
                         std::vector<std::size_t>& out);

        /**
         * @brief Single-sphere visibility test (inline-friendly).
         *
         * Returns true iff the sphere is on the visible side of all 6 planes
         * (with a `radius`-wide slop on each plane — standard conservative test).
         */
        static bool visible(const lfs::core::BlockStore::BlockBounds& b,
                            std::span<const Plane, 6> planes) noexcept;
    };

} // namespace lfs::training::tide
