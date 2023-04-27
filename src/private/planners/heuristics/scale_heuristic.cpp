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


#include "scale_heuristic.hpp"

namespace planners
{
    ScaleHeuristic::ScaleHeuristic(double scalar, HeuristicBase& heuristic) : scalar_(scalar), heuristic_(heuristic) {}

    double ScaleHeuristic::get_cost(const formalism::State& state) const { return scalar_ * heuristic_.get_cost(state); }

    std::vector<double> ScaleHeuristic::get_cost(const std::vector<formalism::State>& states) const
    {
        auto costs = heuristic_.get_cost(states);

        for (auto& cost : costs)
        {
            cost *= scalar_;
        }

        return costs;
    }
}  // namespace planners
