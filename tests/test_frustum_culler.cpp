/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_frustum_culler.cpp
 * @brief Unit tests for tide::FrustumCuller (CPU-only, no CUDA).
 *
 * Constructs synthetic 6-plane frustums and bounding spheres, asserts the
 * culler returns the expected visible-id set. WorkingSet has no CPU-only test
 * because it requires a live CUDA context; integration testing of WorkingSet
 * lives in the trainer end-to-end test (Phase 3).
 */

#include "tide/frustum_culler.hpp"

#include <gtest/gtest.h>

#include <array>

namespace {

    using lfs::core::BlockStore;
    using lfs::training::tide::FrustumCuller;

    // Build a normalized 6-plane axis-aligned box frustum centered on origin
    // spanning [-1, 1] on every axis. Visible side is INSIDE the box.
    std::array<FrustumCuller::Plane, 6> unit_box() {
        return {{
            {+1, 0, 0, 1},  // x >= -1  →  +x + 1 >= 0
            {-1, 0, 0, 1},  // x <=  1  →  -x + 1 >= 0
            {0, +1, 0, 1},
            {0, -1, 0, 1},
            {0, 0, +1, 1},
            {0, 0, -1, 1},
        }};
    }

    BlockStore::BlockBounds sphere(float cx, float cy, float cz, float r) {
        BlockStore::BlockBounds b{};
        b.center[0] = cx;
        b.center[1] = cy;
        b.center[2] = cz;
        b.radius = r;
        return b;
    }

} // namespace

TEST(FrustumCullerTest, SphereInsideBoxIsVisible) {
    auto planes = unit_box();
    EXPECT_TRUE(FrustumCuller::visible(sphere(0, 0, 0, 0.1f),
                                       std::span<const FrustumCuller::Plane, 6>(planes)));
}

TEST(FrustumCullerTest, SphereFullyOutsideBoxIsCulled) {
    auto planes = unit_box();
    // Center at (5,0,0) radius 0.5 → behind +X plane by 3.5 — culled.
    EXPECT_FALSE(FrustumCuller::visible(sphere(5, 0, 0, 0.5f),
                                        std::span<const FrustumCuller::Plane, 6>(planes)));
}

TEST(FrustumCullerTest, SphereStraddlingBoundaryIsVisible) {
    auto planes = unit_box();
    // Center just outside +X plane but radius reaches in.
    EXPECT_TRUE(FrustumCuller::visible(sphere(1.05f, 0, 0, 0.2f),
                                       std::span<const FrustumCuller::Plane, 6>(planes)));
}

TEST(FrustumCullerTest, CullReturnsOnlyVisibleIds) {
    auto planes = unit_box();
    std::vector<BlockStore::BlockBounds> bounds = {
        sphere(0, 0, 0, 0.1f),    // id 0  inside
        sphere(5, 5, 5, 0.1f),    // id 1  far outside
        sphere(-0.5f, 0, 0, 0.1f),// id 2  inside
        sphere(10, 0, 0, 0.5f),   // id 3  far outside
        sphere(0, 0, 0.95f, 0.1f),// id 4  near +Z face but still inside
    };
    std::vector<std::size_t> visible;
    FrustumCuller::cull(bounds,
                        std::span<const FrustumCuller::Plane, 6>(planes),
                        visible);
    EXPECT_EQ(visible.size(), 3u);
    EXPECT_EQ(visible[0], 0u);
    EXPECT_EQ(visible[1], 2u);
    EXPECT_EQ(visible[2], 4u);
}
