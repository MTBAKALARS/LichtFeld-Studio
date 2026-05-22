/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/tide_baker.hpp"

#include "core/block_store.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "io/formats/ply.hpp"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <format>
#include <print>
#include <string>
#include <vector>

namespace lfs::app {

    namespace {

        // Helper: pull a CPU float vector out of a possibly-CUDA tensor.
        std::vector<float> tensor_to_host_vec(const lfs::core::Tensor& t) {
            if (!t.is_valid()) {
                return {};
            }
            return t.cpu().to_vector();
        }

    } // namespace

    int run_tide_baker(const lfs::core::args::TideBakeMode& mode) {
        namespace fs = std::filesystem;

        const auto& ply_path = mode.ply_path;
        const auto& out_dir = mode.out_dir;

        // 1) Validate input.
        if (ply_path.empty() || out_dir.empty()) {
            std::println(stderr, "tide-bake: input PLY and output directory are required");
            return 1;
        }
        std::error_code ec;
        if (!fs::exists(ply_path, ec) || !fs::is_regular_file(ply_path, ec)) {
            std::println(stderr, "tide-bake: input PLY not found: {}",
                         lfs::core::path_to_utf8(ply_path));
            return 1;
        }

        // 2) Handle output directory.
        if (fs::exists(out_dir, ec)) {
            if (!fs::is_directory(out_dir, ec)) {
                std::println(stderr, "tide-bake: output path exists and is not a directory: {}",
                             lfs::core::path_to_utf8(out_dir));
                return 1;
            }
            if (!mode.overwrite) {
                bool empty = true;
                for ([[maybe_unused]] auto& _ : fs::directory_iterator(out_dir, ec)) {
                    empty = false;
                    break;
                }
                if (!empty) {
                    std::println(stderr, "tide-bake: output directory not empty (pass -y/--overwrite to proceed): {}",
                                 lfs::core::path_to_utf8(out_dir));
                    return 1;
                }
            } else {
                fs::remove_all(out_dir, ec);
                fs::create_directories(out_dir, ec);
            }
        } else {
            fs::create_directories(out_dir, ec);
        }

        // 3) Load PLY into SplatData.
        LOG_INFO("tide-bake: loading PLY from {}", lfs::core::path_to_utf8(ply_path));
        const auto t_load_start = std::chrono::steady_clock::now();
        auto splat_res = lfs::io::load_ply(ply_path);
        if (!splat_res) {
            std::println(stderr, "tide-bake: failed to load PLY: {}", splat_res.error());
            return 1;
        }
        auto& splat = *splat_res;
        const std::size_t n_gaussians = splat.size();
        const int sh_degree = splat.get_max_sh_degree();
        const auto t_load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - t_load_start)
                                   .count();
        LOG_INFO("tide-bake: loaded {} Gaussians, SH degree {} ({} ms)",
                 n_gaussians, sh_degree, t_load_ms);

        if (n_gaussians == 0) {
            std::println(stderr, "tide-bake: PLY contains 0 Gaussians");
            return 1;
        }

        // 4) Pull raw tensors to host. PlySource expects raw (pre-activation) values:
        //    means (3), scaling RAW (3, log-space), rotation RAW (4, unnormalized),
        //    opacity RAW (1, pre-sigmoid), sh0 (3), shN (rest_components * 3).
        LOG_INFO("tide-bake: moving tensors to host");
        const auto t_h2_start = std::chrono::steady_clock::now();
        auto means_v = tensor_to_host_vec(splat.means_raw());
        auto scaling_v = tensor_to_host_vec(splat.scaling_raw());
        auto rotation_v = tensor_to_host_vec(splat.rotation_raw());
        auto opacity_v = tensor_to_host_vec(splat.opacity_raw());
        auto sh0_v = tensor_to_host_vec(splat.sh0());
        auto shN_v = tensor_to_host_vec(splat.shN());
        const auto t_h2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - t_h2_start)
                                 .count();
        LOG_INFO("tide-bake: tensors on host ({} ms)", t_h2_ms);

        if (means_v.size() != 3 * n_gaussians) {
            std::println(stderr, "tide-bake: means size mismatch: got {} floats, expected {}",
                         means_v.size(), 3 * n_gaussians);
            return 1;
        }

        // shN may legally be empty (sh_degree == 0). Otherwise it is N * components.
        // PlySource::sh_rest_components is the total floats per Gaussian for shN
        // (e.g. 45 for SH-3 = 15 coefficients * 3 channels).
        std::size_t sh_rest_components = 0;
        if (!shN_v.empty()) {
            const std::size_t per_g_floats = shN_v.size() / n_gaussians;
            if (per_g_floats * n_gaussians != shN_v.size()) {
                std::println(stderr, "tide-bake: shN size {} not divisible by N ({})",
                             shN_v.size(), n_gaussians);
                return 1;
            }
            sh_rest_components = per_g_floats;
        }

        // 5) Configure BlockStore.
        lfs::core::BlockStore::PlySource src;
        src.means = std::span<const float>(means_v);
        src.scaling = std::span<const float>(scaling_v);
        src.rotation = std::span<const float>(rotation_v);
        src.opacity = std::span<const float>(opacity_v);
        src.sh0 = std::span<const float>(sh0_v);
        src.shN = std::span<const float>(shN_v);
        src.num_gaussians = n_gaussians;
        src.sh_rest_components = sh_rest_components;

        lfs::core::BlockStore::Config cfg;
        if (mode.block_size != 0) {
            cfg.block_size = mode.block_size;
        }
        // Phase 3.5.3d: opt-in Adam moments sidecar (manifest v2). Tide
        // out-of-core training with per-block resident Adam requires this.
        cfg.with_moments = mode.with_moments;

        // 6) Stream into BlockStore.
        LOG_INFO("tide-bake: streaming {} Gaussians into BlockStore at {} (block_size={}, with_moments={})",
                 n_gaussians, lfs::core::path_to_utf8(out_dir), cfg.block_size, cfg.with_moments);
        const auto t_bake_start = std::chrono::steady_clock::now();
        auto store_res = lfs::core::BlockStore::stream_ply_to_base(out_dir, src, cfg);
        if (!store_res) {
            std::println(stderr, "tide-bake: stream_ply_to_base failed: {}", store_res.error());
            return 1;
        }
        const auto t_bake_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - t_bake_start)
                                   .count();

        const auto& store = *store_res.value();
        LOG_INFO("tide-bake: done. blocks={} bytes_per_block={} sh_rest_components={} manifest_v{} moments_bpb={} ({} ms)",
                 store.num_blocks(), store.bytes_per_block(), sh_rest_components,
                 store.manifest_version(), store.moments_bytes_per_block(), t_bake_ms);
        std::println("tide-bake OK: {} blocks @ {} B/block (manifest v{}{}) -> {}",
                     store.num_blocks(), store.bytes_per_block(),
                     store.manifest_version(),
                     store.has_moments() ? std::format(", moments {} B/block", store.moments_bytes_per_block()) : std::string{},
                     lfs::core::path_to_utf8(out_dir));
        return 0;
    }

} // namespace lfs::app
