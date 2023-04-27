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


#include "weisfeiler_leman_condition.hpp"

namespace planners
{
    WeisfeilerLemanCondition::WeisfeilerLemanCondition(const formalism::ProblemDescription& problem) : problem_(problem), weisfeiler_leman_(1), colors_() {}

    // Returns true if an 1-WL isomorphic state has been seen before, otherwise false
    bool WeisfeilerLemanCondition::test(const formalism::State& state)
    {
        const auto color = weisfeiler_leman_.compute_state_color(problem_, state);
        return !colors_.insert(color).second;
    }
}  // namespace planners
