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


#include "optimal_reward_function.hpp"

namespace experiments
{
    OptimalRewardFunctionImpl::OptimalRewardFunctionImpl() : optimal_value_function_() {}

    double OptimalRewardFunctionImpl::cumulative_optimal_reward(const planners::StateSpace& state_space, const formalism::State& state)
    {
        return (double) optimal_value_function_.get_value(state_space, state);
    }

    double OptimalRewardFunctionImpl::reward(const planners::StateSpace& state_space, const formalism::State& from_state, const formalism::State& to_state)
    {
        const auto from_reward = (double) optimal_value_function_.get_value(state_space, from_state);
        const auto to_reward = (double) optimal_value_function_.get_value(state_space, to_state);
        return from_reward - to_reward;
    }

    OptimalRewardFunction create_optimal_reward_function() { return std::make_shared<OptimalRewardFunctionImpl>(); }
}  // namespace experiments
