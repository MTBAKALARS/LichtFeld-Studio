/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include "core/parameters.hpp"
#include <expected>
#include <filesystem>
#include <memory>
#include <variant>

namespace lfs::core::args {

    // Parsed argument modes
    struct TrainingMode {
        std::unique_ptr<param::TrainingParameters> params;
    };
    struct ConvertMode {
        param::ConvertParameters params;
    };
    struct HelpMode {};
    struct VersionMode {};
    struct WarmupMode {}; // JIT compile PTX kernels and exit

    /// Bake an existing .ply into a Tide BlockStore on disk (Phase 3.4b).
    /// Triggered by the `tide-bake` subcommand. The resulting directory can
    /// then be passed to `--tide-store` for an out-of-core training run.
    struct TideBakeMode {
        std::filesystem::path ply_path;
        std::filesystem::path out_dir;
        std::size_t block_size = 0;  ///< 0 = BlockStore::kDefaultBlockSize
        bool overwrite = false;
        /// Phase 3.5.3d. When true, the baker writes a manifest-v2 store with
        /// a zero-initialized `moments.bin` sidecar sized for per-block Adam
        /// state (m + v, 472 B per Gaussian). Required for Tide out-of-core
        /// training with per-block resident Adam (Phase 3.5.3e+).
        bool with_moments = false;
    };
    struct PluginMode {
        enum class Command { CREATE,
                             CHECK,
                             LIST };
        Command command;
        std::string name;
    };

    using ParsedArgs = std::variant<TrainingMode, ConvertMode, HelpMode, VersionMode, WarmupMode, PluginMode, TideBakeMode>;

    LFS_CORE_API std::expected<ParsedArgs, std::string> parse_args(int argc, const char* const argv[]);

    // Legacy interface - prefer parse_args()
    LFS_CORE_API std::expected<std::unique_ptr<param::TrainingParameters>, std::string>
    parse_args_and_params(int argc, const char* const argv[]);

} // namespace lfs::core::args
