/*
 * Copyright (C) 2023 Simon Stahlberg
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */


#include "cumulative_return_function.hpp"
#include "lambda_return_function.hpp"
#include "n_step_return_function.hpp"

#include <memory>

namespace experiments
{
    LambdaReturnFunctionImpl::LambdaReturnFunctionImpl(double lambda) : lambda_(lambda) {}

    torch::Tensor
    LambdaReturnFunctionImpl::get(int32_t step_index, int32_t num_steps, double discount, const Trajectory& trajectory, const torch::Device device) const
    {
        const NStepReturnFunctionImpl n_step_return;
        const CumulativeReturnFunctionImpl cumulative_return;

        // TODO: Optimize this, a lot of duplicate calculations are done by calling n_step_return multiple times

        const auto remaining_steps = (int32_t) trajectory.size() - step_index;

        auto cumulative_lambda_advantage = torch::zeros(1, device);
        auto cumulative_lambda = 1.0;

        for (int32_t num_steps = 1; num_steps <= remaining_steps; ++num_steps)
        {
            cumulative_lambda_advantage += cumulative_lambda * n_step_return.get(step_index, num_steps, discount, trajectory, device);
            cumulative_lambda *= lambda_;
        }

        return (1.0 - lambda_) * cumulative_lambda_advantage
               + cumulative_lambda * cumulative_return.get(step_index, remaining_steps, discount, trajectory, device);
    }
}  // namespace experiments
