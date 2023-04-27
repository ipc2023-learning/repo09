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


#include "value_decomposition_search.hpp"

namespace planners
{
    bool decomposition_search(const formalism::ProblemDescription& problem,
                              const std::function<bool(const formalism::State&, formalism::ActionList&)>& planner,
                              formalism::ActionList& out_plan)
    {
        out_plan.clear();
        formalism::State state = problem->initial;

        while (!formalism::literals_hold(problem->goal, state))
        {
            formalism::ActionList decomposed_plan;
            const auto found_plan = planner(state, decomposed_plan);

            if (!found_plan)
            {
                return false;
            }

            for (const auto& action : decomposed_plan)
            {
                if (!formalism::is_applicable(action, state))
                {
                    std::cerr << "[Error] Action not applicable on state" << std::endl;
                    std::cerr << state << std::endl;
                    std::cerr << action << std::endl;
                    throw std::runtime_error("bug: plan not applicable");
                }

                state = formalism::apply(action, state);
            }

            out_plan.insert(out_plan.end(), decomposed_plan.begin(), decomposed_plan.end());
        }

        return true;
    }
}  // namespace planners
