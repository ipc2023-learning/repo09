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


#include "../../formalism/action.hpp"
#include "../../formalism/action_schema.hpp"
#include "../../formalism/atom.hpp"
#include "../../formalism/literal.hpp"
#include "../../formalism/problem.hpp"
#include "../../formalism/state.hpp"
#include "applicable_actions_naive_generator.hpp"

#include <unordered_set>

namespace planners
{
    ApplicableActionsNaiveGenerator::ApplicableActionsNaiveGenerator(const formalism::ActionSchema& action_schema,
                                                                     const formalism::ProblemDescription& problem) :
        action_schema(action_schema),
        objects_by_parameter_type()
    {
        for (const auto& parameter : action_schema->parameters)
        {
            std::unordered_set<formalism::Object> compatible_objects;

            for (const auto& object : problem->objects)
            {
                if (formalism::is_subtype_of(object->type, parameter->type))
                {
                    compatible_objects.insert(object);
                }
            }

            objects_by_parameter_type.insert(std::make_pair(parameter, compatible_objects));
        }
    }

    std::vector<formalism::ParameterAssignment> ApplicableActionsNaiveGenerator::get_assignments(const ParameterAssignmentsMap::const_iterator iterator,
                                                                                                 const ParameterAssignmentsMap::const_iterator end) const
    {
        std::vector<formalism::ParameterAssignment> assignments;

        if (iterator != end)
        {
            const auto sub_assignments = get_assignments(std::next(iterator), end);
            const auto parameter = iterator->first;

            for (const auto& sub_assignment : sub_assignments)
            {
                for (const auto& object : iterator->second)
                {
                    auto assignment = sub_assignment;
                    assignment.insert(std::make_pair(parameter, object));
                    assignments.push_back(assignment);
                }
            }
        }
        else
        {
            const formalism::ParameterAssignment empty_assignment;
            assignments.push_back(empty_assignment);
        }

        return assignments;
    }

    std::vector<formalism::Action> ApplicableActionsNaiveGenerator::get_applicable_actions(const formalism::State& state) const
    {
        std::vector<formalism::Action> applicable_actions;

        const auto assignments = get_assignments(objects_by_parameter_type.begin(), objects_by_parameter_type.end());

        for (const auto& assignment : assignments)
        {
            const auto ground_action = formalism::create_action(state->get_problem(), action_schema, assignment);

            if (formalism::literals_hold(ground_action->get_precondition(), state))
            {
                applicable_actions.push_back(ground_action);
            }
        }

        return applicable_actions;
    }
}  // namespace planners
