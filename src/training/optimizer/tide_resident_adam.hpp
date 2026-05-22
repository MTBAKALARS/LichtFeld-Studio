/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "adam_optimizer.hpp" // For AdamConfig, ParamType

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>

namespace lfs::training::tide {

    /**
     * @brief Adam optimizer for the Tide working-set path.
     *
     * Unlike @ref lfs::training::AdamOptimizer, this class does **not** own the
     * parameter tensors — it operates on raw device pointers supplied by the
     * caller (eventually @ref TideStrategy, pointing into a @ref WorkingSet active
     * buffer). The Adam state (exp_avg, exp_avg_sq) IS owned here and is sized to
     * cover the working-set capacity, not the global gaussian count.
     *
     * The class mirrors @ref AdamOptimizer's per-@ref ParamType bookkeeping
     * (separate step_count and learning rate per param type) and reuses the
     * exact same CUDA kernel (`fast_lfs::optimizer::adam_step_raw`), so produced
     * updates are numerically equivalent to AdamOptimizer when fed the same
     * inputs.
     *
     * **Phase 3.1 scope (this commit)**: standalone math — fixed-size state
     * buffers, no working-set integration, no per-block bookkeeping. Phase 3.2
     * adds the per-block routing/retention dance and BlockStore writeback.
     *
     * Thread-safety: not safe for concurrent calls. Caller (the trainer) drives
     * it from a single thread per CUDA stream.
     */
    class TideResidentAdam {
    public:
        /// Per-(ParamType) sizing supplied at create() time. One entry per
        /// param type the trainer intends to step (typically all six).
        struct ParamSpec {
            ParamType type = ParamType::Means;
            std::size_t num_elements = 0; ///< Total floats in the contiguous device buffer for this param
        };

        struct Config {
            AdamConfig adam;                 ///< Must match the AdamConfig the legacy path would use
            int cuda_device = 0;             ///< GPU index for cudaSetDevice
            int sh_warmup_iterations = 1000; ///< Iters <= this skip ShN step (matches AdamOptimizer)
        };

        struct Stats {
            std::uint64_t steps_total = 0;     ///< Number of successful step() calls across all param types
            std::uint64_t steps_skipped = 0;   ///< step() calls that were no-ops (e.g. ShN during warmup)
        };

        ~TideResidentAdam();
        TideResidentAdam(const TideResidentAdam&) = delete;
        TideResidentAdam& operator=(const TideResidentAdam&) = delete;
        TideResidentAdam(TideResidentAdam&&) noexcept;
        TideResidentAdam& operator=(TideResidentAdam&&) noexcept;

        /**
         * @brief Allocate exp_avg/exp_avg_sq state buffers for the requested params.
         *
         * For each entry in @p params an internal Adam state pair is allocated of
         * exactly @ref ParamSpec::num_elements floats and zero-initialized.
         * Subsequent @ref step() calls must use the same num_elements per type.
         *
         * @return TideResidentAdam on success; descriptive error on allocation
         *         failure or duplicate ParamType in @p params.
         */
        static std::expected<std::unique_ptr<TideResidentAdam>, std::string>
        create(const Config& config, std::span<const ParamSpec> params);

        /**
         * @brief Perform one Adam step on a single param type's buffer.
         *
         * - Increments step_count for @p type.
         * - Skips ShN entirely while @p iteration <= sh_warmup_iterations.
         * - Calls @c fast_lfs::optimizer::adam_step_raw with the per-param LR,
         *   the global beta1/beta2/eps, and freshly-computed bias corrections.
         *
         * The caller must guarantee that @p num_elements matches the value passed
         * to @ref create() for the same @p type; otherwise an error is returned
         * without touching device memory.
         *
         * @param type Which param to step.
         * @param param Device pointer to the param buffer (writable, num_elements floats).
         * @param grad Device pointer to the gradient buffer (read-only, num_elements floats).
         * @param num_elements Length of the param/grad buffers in floats.
         * @param iteration Current training iteration (>= 1).
         */
        std::expected<void, std::string>
        step(ParamType type, float* param, const float* grad,
             std::size_t num_elements, int iteration);

        /**
         * @brief Zero exp_avg / exp_avg_sq for a param type and reset its step_count.
         *
         * Useful when the working set evicts all resident state and starts fresh.
         */
        std::expected<void, std::string> reset(ParamType type);

        /// True if state was allocated for this ParamType in create().
        bool has_param(ParamType type) const noexcept;

        /// Per-type step counter, monotonically increasing across step() calls.
        int64_t step_count(ParamType type) const noexcept;

        /// Read-only device pointer to the first-moment buffer for this param.
        /// Returns nullptr if has_param(type) is false. Lifetime tied to *this*.
        const float* exp_avg(ParamType type) const noexcept;

        /// Read-only device pointer to the second-moment buffer for this param.
        const float* exp_avg_sq(ParamType type) const noexcept;

        const Config& config() const noexcept { return config_; }
        const Stats& stats() const noexcept { return stats_; }

    private:
        TideResidentAdam() = default;

        struct Impl;
        std::unique_ptr<Impl> impl_;

        Config config_{};
        Stats stats_{};
    };

} // namespace lfs::training::tide
