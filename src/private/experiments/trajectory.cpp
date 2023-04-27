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


#include "trajectory.hpp"

namespace experiments
{
    TrajectoryStep::TrajectoryStep(const planners::StateSpace& state_space,
                                   const formalism::State& state,
                                   const formalism::State& successor_state,
                                   const torch::Tensor& transition_probability,
                                   const torch::Tensor& state_value,
                                   const torch::Tensor& successor_value,
                                   const double reward) :
        state_space(state_space),
        state(state),
        successor_state(successor_state),
        transition_probability(transition_probability),
        state_value(state_value),
        successor_value(successor_value),
        reward(reward)
    {
    }
}  // namespace experiments
