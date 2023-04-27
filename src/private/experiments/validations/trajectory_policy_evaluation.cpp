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


#include "trajectory_policy_evaluation.hpp"

namespace experiments
{
    double TrajectoryPolicyEvaluation::initialize(const planners::StateSpaceList& state_spaces, models::RelationalNeuralNetwork& model)
    {
        throw std::runtime_error("not implemented");
    }

    double TrajectoryPolicyEvaluation::evaluate(models::RelationalNeuralNetwork& model) { throw std::runtime_error("not implemented"); }
}  // namespace experiments
