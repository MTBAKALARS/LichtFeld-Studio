/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "optimizer/render_output.hpp"
#include <expected>
#include <filesystem>
#include <istream>
#include <memory>
#include <optional>
#include <ostream>
#include <string>

namespace lfs::io {
    class PipelinedImageLoader;
}

namespace lfs::core {
    class Camera;
}

namespace lfs::training {

    class CameraDataset;

    /**
     * @brief Strategy interface for Gaussian splatting optimization.
     *
     * Strategies operate on a SplatData reference owned by the Scene.
     * This allows the same model to be used for both training and visualization.
     */
    class IStrategy {
    public:
        virtual ~IStrategy() = default;

        virtual void initialize(const lfs::core::param::OptimizationParameters& optimParams) = 0;

        /// Phase 3.5: hook called by the trainer immediately BEFORE the
        /// per-iteration rasterizer forward pass, after the current camera
        /// has been selected. Default no-op for strategies that don't need
        /// camera-aware setup. Tide-class strategies override this to
        /// frustum-cull blocks and ensure the resident set covers the visible
        /// region (issuing async prefetches and `wait_and_activate` on the
        /// WorkingSet).
        ///
        /// Contract: callers MUST invoke this between camera selection and
        /// rasterize_forward. Implementations MUST be safe to call every
        /// iteration; they may issue synchronous loads on the first call and
        /// short-circuit on subsequent calls if the resident set already
        /// covers the camera.
        virtual void pre_forward(int /*iter*/, const lfs::core::Camera& /*cam*/) {}

        /// Optional pipelining hook. Strategies that overlap streaming work with
        /// training (e.g. TideStrategy) may stage the resident set predicted for
        /// the next iteration's camera onto an inactive WorkingSet buffer on a
        /// dedicated CUDA stream. Trainers may call this at end-of-iter once
        /// they have peeked the next camera; the matching `pre_forward(next_iter,
        /// next_cam)` at the start of the next iteration consumes it via
        /// `wait_and_activate`. Safe to call repeatedly or never; default no-op.
        virtual void prefetch_next(int /*next_iter*/, const lfs::core::Camera& /*next_cam*/) {}

        virtual void pre_step(int /*iter*/, RenderOutput& /*render_output*/) {}

        virtual void post_backward(int iter, RenderOutput& render_output) = 0;

        virtual void step(int iter) = 0;

        virtual bool is_refining(int iter) const = 0;

        // Get the underlying Gaussian model (reference to Scene-owned data)
        virtual lfs::core::SplatData& get_model() = 0;
        virtual const lfs::core::SplatData& get_model() const = 0;

        // Get the optimizer (for gradient access during backward pass)
        virtual class AdamOptimizer& get_optimizer() = 0;
        virtual const class AdamOptimizer& get_optimizer() const = 0;

        // Remove Gaussians based on mask
        virtual void remove_gaussians(const lfs::core::Tensor& mask) = 0;

        // Serialization for checkpoints
        virtual void serialize(std::ostream& os) const = 0;
        virtual void deserialize(std::istream& is) = 0;

        // Strategy type identifier for checkpoint compatibility
        virtual const char* strategy_type() const = 0;

        // Reserve optimizer capacity for future growth (e.g., after checkpoint load)
        virtual void reserve_optimizer_capacity(size_t capacity) = 0;

        // Update the strategy's cached optimization parameters after checkpoint params are resolved.
        virtual void set_optimization_params(const lfs::core::param::OptimizationParameters&) {}

        // Optional hook for strategies that need the training dataset (e.g., for view-based scoring)
        virtual void set_training_dataset(std::shared_ptr<CameraDataset>) {}

        virtual void set_image_loader(lfs::io::PipelinedImageLoader*) {}

        /// Phase 3.5.9 Plan A: optional hook for strategies whose @ref get_model()
        /// represents only a subset of the strategy's full trainable state (e.g.
        /// @ref TideStrategy::get_model returns just the resident WorkingSet of a
        /// much larger on-disk BlockStore — typically <25% of the full model on a
        /// 24 GB GPU at 30M+SH-3). The default implementation returns
        /// std::nullopt, signaling "no override — use the standard
        /// lfs::io::save_ply(get_model(), opts) path". Strategies that override
        /// this MUST write a binary 3DGS PLY containing every Gaussian in their
        /// full model to @p output_path and return the result. The trainer's
        /// @ref Trainer::save_ply call dispatches through this hook before
        /// falling back to the standard path.
        virtual std::optional<std::expected<void, std::string>>
        save_full_ply(const std::filesystem::path& /*output_path*/, bool /*binary*/) {
            return std::nullopt;
        }
    };
} // namespace lfs::training
