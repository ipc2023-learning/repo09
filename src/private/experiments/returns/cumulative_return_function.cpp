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

#include <memory>

namespace experiments
{
    torch::Tensor
    CumulativeReturnFunctionImpl::get(int32_t step_index, int32_t num_steps, double discount, const Trajectory& trajectory, const torch::Device device) const
    {
        const auto max_index = std::min(step_index + num_steps, (int32_t) trajectory.size());

        auto cumulative_return = 0.0;
        auto cumulative_discount = 1.0;

        for (; step_index < max_index; ++step_index)
        {
            cumulative_return += cumulative_discount * trajectory.at(step_index).reward;
            cumulative_discount *= discount;
        }

        return torch::scalar_tensor(cumulative_return, device);
    }
}  // namespace experiments
