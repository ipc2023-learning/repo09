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


#include "batched_bfs_search.hpp"

#include "search_base.hpp"

#include <algorithm>
#include <chrono>
#include <deque>
#include <queue>
#include <unordered_set>

namespace planners
{
    bool batched_bfs_search(const formalism::State& initial_state,
                            planners::SuccessorGenerator& successor_generator,
                            planners::HeuristicBase& heuristic,
                            planners::ConditionBase& goal,
                            planners::ConditionBase& prune,
                            SearchStatistics& statistics,
                            formalism::ActionList& out_plan)
    {
        statistics.reset();

        if (goal.test(initial_state))
        {
            out_plan.clear();
            statistics.time_end = std::chrono::high_resolution_clock::now();
            return true;
        }

        if (prune.test(initial_state))
        {
            statistics.time_end = std::chrono::high_resolution_clock::now();
            return false;
        }

        // const auto initial_heuristic = heuristic.get_cost(initial_state);
        uint32_t initial_state_index;
        SearchStateRepository state_repository;
        state_repository.add_or_get_state(initial_state, initial_state_index);

        std::vector<uint32_t> fringe;
        fringe.emplace_back(initial_state_index);
        auto found_solution = false;

        while (fringe.size() > 0)
        {
            std::vector<formalism::State> successor_states;
            std::vector<uint32_t> successor_state_indices;

            for (const auto repository_index : fringe)
            {
                ++statistics.num_expanded;
                const auto state = state_repository.get_state(repository_index);

                const auto time_start = std::chrono::high_resolution_clock::now();
                const auto applicable_actions = successor_generator->get_applicable_actions(state);
                const auto time_end = std::chrono::high_resolution_clock::now();
                statistics.duration_successor_generator_seconds += std::chrono::duration_cast<std::chrono::seconds>(time_end - time_start);

                for (const auto& action : applicable_actions)
                {
                    ++statistics.num_generated;
                    const auto successor_state = formalism::apply(action, state);
                    uint32_t successor_repository_index;

                    const auto successor_prune = prune.test(successor_state);

                    if (!successor_prune && state_repository.add_or_get_state(successor_state, successor_repository_index))
                    {
                        auto& context = state_repository.get_context(successor_repository_index);
                        context.predecessor_action = action;
                        context.predecessor_state_index = repository_index;

                        successor_states.push_back(successor_state);
                        successor_state_indices.push_back(successor_repository_index);
                    }
                }
            }

            const auto time_start = std::chrono::high_resolution_clock::now();
            const auto successor_heuristics = heuristic.get_cost(successor_states);
            const auto time_end = std::chrono::high_resolution_clock::now();
            statistics.duration_heuristic_seconds += std::chrono::duration_cast<std::chrono::seconds>(time_end - time_start);

            fringe.clear();

            for (std::size_t successor_index = 0; successor_index < successor_heuristics.size(); ++successor_index)
            {
                const auto successor_state = successor_states[successor_index];
                const auto successor_repository_index = successor_state_indices[successor_index];

                if (goal.test(successor_state))
                {
                    const auto found_plan = state_repository.get_plan(successor_repository_index);
                    found_solution = true;
                    out_plan.clear();
                    out_plan.insert(out_plan.end(), found_plan.begin(), found_plan.end());
                    break;
                }

                fringe.emplace_back(successor_repository_index);
            }

            if (found_solution)
            {
                break;
            }
        }

        statistics.time_end = std::chrono::high_resolution_clock::now();
        return found_solution;
    }
}  // namespace planners
