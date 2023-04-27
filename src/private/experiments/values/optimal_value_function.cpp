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


#include "../../planners/generators/lifted_schema_successor_generator.hpp"
#include "optimal_value_function.hpp"

#include <deque>
#include <limits>
#include <unordered_map>

namespace experiments
{
    OptimalValueFunction::OptimalValueFunction() {}

    uint32_t OptimalValueFunction::get_value(const planners::StateSpace& state_space, const formalism::State& state)
    {
        return (uint32_t) state_space->get_distance_to_goal_state(state);
    }
}  // namespace experiments
