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


#include "fixed_horizon_reinforce_baseline.hpp"
#include "returns/cumulative_return_function.hpp"

namespace experiments
{
    FixedHorizonReinforceBaseline::FixedHorizonReinforceBaseline(uint32_t batch_size,
                                                                 uint32_t max_epochs,
                                                                 double learning_rate,
                                                                 uint32_t max_horizon,
                                                                 double discount,
                                                                 double bounds_factor,
                                                                 const RewardFunction& reward_function) :
        FixedHorizonValueBase(batch_size, max_epochs, learning_rate, max_horizon, discount, bounds_factor, reward_function)
    {
    }

    torch::Tensor FixedHorizonReinforceBaseline::cumulative_return(int32_t step_index, const Trajectory& trajectory) const
    {
        const auto& step = trajectory[step_index];
        const auto& device = step.state_value.device();
        const auto max_index = (int32_t) trajectory.size() - step_index;
        const auto discount = get_discount();

        auto cumulative_return = 0.0;
        auto cumulative_discount = 1.0;

        for (; step_index < max_index; ++step_index)
        {
            cumulative_return += cumulative_discount * trajectory.at(step_index).reward;
            cumulative_discount *= discount;
        }

        return torch::scalar_tensor(cumulative_return, device);
    }

    torch::Tensor FixedHorizonReinforceBaseline::get_update_value(const formalism::ProblemDescription& problem,
                                                                  const experiments::Trajectory& trajectory,
                                                                  std::size_t step_index) const
    {
        const auto& step = trajectory[step_index];
        const auto g = cumulative_return(step_index, trajectory);
        const auto state_value = step.state_value.detach()[0];
        const auto update_value = g - state_value;
        return update_value;
    }
}  // namespace experiments
