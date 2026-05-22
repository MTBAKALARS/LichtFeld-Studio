/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file tide_baker.hpp
 * @brief Phase 3.4b — `tide-bake` subcommand: convert an existing .ply into
 *        a Tide BlockStore directory on disk.
 *
 * The resulting directory can be passed to `--tide-store=<path>` when
 * launching a training run with `--strategy=tide`.
 */

#include "core/argument_parser.hpp"

namespace lfs::app {

    /// Run the `tide-bake` subcommand. Returns 0 on success, non-zero on failure.
    int run_tide_baker(const lfs::core::args::TideBakeMode& mode);

} // namespace lfs::app
