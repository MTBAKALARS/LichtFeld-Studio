/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_tide_bake_args.cpp
 * @brief Phase 3.5.3d unit tests for the `tide-bake` CLI argument parser.
 *
 * Verifies that the `--with-moments` flag (and the existing options it sits
 * alongside) parse into a TideBakeMode with the expected fields set. These
 * are pure parser tests; the end-to-end Adam-moments sidecar bake path is
 * covered by BlockStoreMomentsTest (test_block_store.cpp) and
 * WorkingSetMomentsTest (test_working_set.cpp).
 */

#include "core/argument_parser.hpp"

#include <gtest/gtest.h>

#include <array>
#include <string>
#include <variant>

namespace {

    using lfs::core::args::parse_args;
    using lfs::core::args::TideBakeMode;

    // Helper: build a (argc, argv) pair from a std::array of C strings. The
    // backing array lives for the test's scope.
    template <std::size_t N>
    int parse_into(const std::array<const char*, N>& argv,
                   TideBakeMode& out) {
        auto r = parse_args(static_cast<int>(N), argv.data());
        if (!r) {
            ADD_FAILURE() << "parse_args failed: " << r.error();
            return 1;
        }
        if (!std::holds_alternative<TideBakeMode>(*r)) {
            ADD_FAILURE() << "parse_args did not return a TideBakeMode";
            return 2;
        }
        out = std::get<TideBakeMode>(*r);
        return 0;
    }

    // -----------------------------------------------------------------------
    // Phase 3.5.3d — `--with-moments` flag
    // -----------------------------------------------------------------------

    TEST(TideBakeArgsTest, DefaultsWithMomentsFalse) {
        std::array<const char*, 4> argv{
            "LichtFeld-Studio", "tide-bake", "in.ply", "out_dir"};
        TideBakeMode mode;
        ASSERT_EQ(parse_into(argv, mode), 0);
        EXPECT_FALSE(mode.with_moments);
        EXPECT_FALSE(mode.overwrite);
        EXPECT_EQ(mode.block_size, 0u); // 0 = BlockStore default
        EXPECT_EQ(mode.ply_path.filename(), "in.ply");
        EXPECT_EQ(mode.out_dir.filename(), "out_dir");
    }

    TEST(TideBakeArgsTest, WithMomentsFlagSetsBit) {
        std::array<const char*, 5> argv{
            "LichtFeld-Studio", "tide-bake", "in.ply", "out_dir", "--with-moments"};
        TideBakeMode mode;
        ASSERT_EQ(parse_into(argv, mode), 0);
        EXPECT_TRUE(mode.with_moments);
        // Other flags untouched.
        EXPECT_FALSE(mode.overwrite);
        EXPECT_EQ(mode.block_size, 0u);
    }

    TEST(TideBakeArgsTest, WithMomentsComposesWithOtherFlags) {
        // Order is irrelevant: --with-moments + --overwrite + --block-size=N.
        std::array<const char*, 7> argv{
            "LichtFeld-Studio", "tide-bake",
            "--with-moments", "--overwrite", "--block-size=8192",
            "in.ply", "out_dir"};
        TideBakeMode mode;
        ASSERT_EQ(parse_into(argv, mode), 0);
        EXPECT_TRUE(mode.with_moments);
        EXPECT_TRUE(mode.overwrite);
        EXPECT_EQ(mode.block_size, 8192u);
        EXPECT_EQ(mode.ply_path.filename(), "in.ply");
        EXPECT_EQ(mode.out_dir.filename(), "out_dir");
    }

    TEST(TideBakeArgsTest, UnknownFlagStillRejected) {
        // Sanity: the parser still rejects unknown flags even after the new
        // option landed (no greedy-substring fall-through).
        std::array<const char*, 5> argv{
            "LichtFeld-Studio", "tide-bake", "in.ply", "out_dir", "--with-momentss"};
        auto r = parse_args(static_cast<int>(argv.size()), argv.data());
        ASSERT_FALSE(r.has_value());
        EXPECT_NE(r.error().find("Unknown tide-bake option"), std::string::npos)
            << "actual error: " << r.error();
    }

} // namespace
