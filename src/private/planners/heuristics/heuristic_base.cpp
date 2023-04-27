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


#include "../../formalism/state.hpp"
#include "heuristic_base.hpp"

#include <vector>

namespace planners
{
    std::vector<double> HeuristicBase::get_cost(const std::vector<formalism::State>& states) const
    {
        std::vector<double> values;

        for (const auto& state : states)
        {
            values.emplace_back(get_cost(state));
        }

        return values;
    }

}  // namespace planners
