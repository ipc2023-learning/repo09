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


#include "fixed_horizon_value_base.hpp"

namespace experiments
{
    double FixedHorizonValueBase::get_bounds_factor() const { return bounds_factor_; }

    torch::Tensor FixedHorizonValueBase::update_value_gradient(const torch::Tensor gradient,
                                                               const torch::Tensor update_value,
                                                               const experiments::TrajectoryStep& step,
                                                               const double cumulative_discount) const
    {
        auto updated_gradient = gradient;
        updated_gradient += update_value * step.state_value[0];

        if (formalism::literals_hold(step.state_space->problem->goal, step.successor_state))
        {
            // The value of goal states is 0
            updated_gradient -= step.successor_value[0].abs();
        }

        // RL does not typically work very well when rewards are sparse.
        // We use upper/bigger and lower/smaller bounds to help shape the value function without
        // adding an optimality constraint.

        const auto bounds_factor = get_bounds_factor();

        if (bounds_factor >= 1.0)
        {
            auto reward_function = get_reward_function();
            const auto optimal_reward = reward_function->cumulative_optimal_reward(step.state_space, step.state);
            const auto discount = get_discount();
            const auto state_value = step.state_value.detach()[0];
            const auto sign = std::signbit(optimal_reward) ? -1 : 1;
            const auto label_return = sign * (1.0 - std::pow(discount, std::abs(optimal_reward))) / (1.0 - discount);
            const auto label_bigger_bound = torch::clamp_min(bounds_factor * label_return - state_value, 0.0);
            const auto label_smaller_bound = torch::clamp_max(label_return - state_value, 0.0);

            updated_gradient += label_bigger_bound * step.state_value[0];
            updated_gradient += label_smaller_bound * step.state_value[0];
        }

        return updated_gradient;
    }

    FixedHorizonValueBase::FixedHorizonValueBase(uint32_t batch_size,
                                                 uint32_t max_epochs,
                                                 double learning_rate,
                                                 uint32_t max_horizon,
                                                 double discount,
                                                 double bounds_factor,
                                                 const RewardFunction& reward_function) :
        FixedHorizonPolicyGradient(batch_size, max_epochs, learning_rate, max_horizon, discount, reward_function),
        bounds_factor_(bounds_factor)
    {
    }
}  // namespace experiments
