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


#include "batched_astar_search.hpp"

#include <algorithm>
#include <chrono>
#include <deque>
#include <functional>
#include <limits>
#include <queue>
#include <unordered_set>

namespace planners
{
    bool batched_astar_search(const BatchedAstarSettings& settings,
                              const formalism::State& initial_state,
                              planners::SearchStateRepository& state_repository,
                              planners::SuccessorGenerator& successor_generator,
                              planners::HeuristicBase& heuristic,
                              planners::ConditionBase& goal_condition,
                              planners::ConditionBase& prune_condition,
                              SearchStatistics& statistics,
                              std::chrono::high_resolution_clock::time_point& time_end,
                              formalism::ActionList& output_plan)
    {
        statistics.reset();

        // Check if initial state is either a goal state or pruned

        if (goal_condition.test(initial_state))
        {
            output_plan.clear();
            statistics.time_end = std::chrono::high_resolution_clock::now();
            return true;
        }

        if (prune_condition.test(initial_state))
        {
            statistics.time_end = std::chrono::high_resolution_clock::now();
            return false;
        }

        // Setup initial state

        const auto initial_heuristic_value = heuristic.get_cost(initial_state);
        uint32_t initial_repository_index;
        state_repository.add_or_get_state(initial_state, initial_repository_index);
        auto& initial_context = state_repository.get_context(initial_repository_index);
        initial_context.cost = 0.0;
        initial_context.heuristic_value = initial_heuristic_value;
        initial_context.predecessor_action = nullptr;
        initial_context.predecessor_state_index = -1U;

        // Setup priority queue

        const auto min_f_value = [&state_repository](uint32_t lhs, uint32_t rhs)
        {
            const auto& lhs_context = state_repository.get_context(lhs);
            const auto& rhs_context = state_repository.get_context(rhs);

            const auto lhs_f_value = (lhs_context.cost + lhs_context.heuristic_value);
            const auto rhs_f_value = (rhs_context.cost + rhs_context.heuristic_value);

            return lhs_f_value > rhs_f_value;
        };

        std::priority_queue<uint32_t, std::vector<uint32_t>, std::function<bool(uint32_t, uint32_t)>> open_list(min_f_value);
        open_list.emplace(initial_repository_index);

        // Setup solution

        auto has_solution = false;
        auto solution_cost = std::numeric_limits<double>::infinity();
        auto solution_repository_index = 0U;
        auto improvement_found = false;

        // Start search

        while (!open_list.empty() && (!improvement_found || (statistics.num_expanded < settings.min_expanded))
               && (statistics.num_expanded < settings.max_expanded))
        {
            if (std::chrono::high_resolution_clock::now() > time_end)
            {
                break;
            }

            std::vector<formalism::State> successor_states;
            std::vector<uint32_t> successor_repository_indices;

            const auto& top_context = state_repository.get_context(open_list.top());
            const auto max_f_value = top_context.cost + top_context.heuristic_value + settings.batch_delta;

            while (!open_list.empty() && (successor_states.size() < settings.batch_size))
            {
                const auto repository_index = open_list.top();
                const auto& state_context = state_repository.get_context(repository_index);
                const auto f_value = state_context.cost + state_context.heuristic_value;

                if (f_value < max_f_value)
                {
                    open_list.pop();

                    ++statistics.num_expanded;
                    const auto& state = state_repository.get_state(repository_index);

                    const auto time_start = std::chrono::high_resolution_clock::now();
                    const auto applicable_actions = successor_generator->get_applicable_actions(state);
                    const auto time_end = std::chrono::high_resolution_clock::now();
                    statistics.duration_successor_generator_seconds += std::chrono::duration_cast<std::chrono::seconds>(time_end - time_start);

                    for (const auto& action : applicable_actions)
                    {
                        ++statistics.num_generated;
                        const auto successor_state = formalism::apply(action, state);
                        uint32_t successor_repository_index;

                        const auto successor_prune = prune_condition.test(successor_state);

                        if (!successor_prune && state_repository.add_or_get_state(successor_state, successor_repository_index))
                        {
                            auto& successor_context = state_repository.get_context(successor_repository_index);
                            successor_context.cost = state_context.cost + action->cost;
                            successor_context.predecessor_action = action;
                            successor_context.predecessor_state_index = repository_index;

                            successor_states.emplace_back(successor_state);
                            successor_repository_indices.emplace_back(successor_repository_index);
                        }
                    }
                }
                else
                {
                    break;
                }
            }

            const auto time_start = std::chrono::high_resolution_clock::now();
            const auto successor_heuristics = heuristic.get_cost(successor_states);
            const auto time_end = std::chrono::high_resolution_clock::now();
            statistics.duration_heuristic_seconds += std::chrono::duration_cast<std::chrono::seconds>(time_end - time_start);

            for (std::size_t index = 0; index < successor_heuristics.size(); ++index)
            {
                const auto successor_repository_index = successor_repository_indices[index];
                const auto successor_heuristic = successor_heuristics[index];
                const auto successor_state = successor_states[index];
                const auto successor_goal = goal_condition.test(successor_state);

                auto& successor_context = state_repository.get_context(successor_repository_index);
                successor_context.heuristic_value = successor_heuristic;

                if (successor_goal)
                {
                    has_solution = true;
                    const auto successor_cost = successor_context.cost;

                    if (successor_cost < solution_cost)
                    {
                        solution_cost = successor_cost;
                        solution_repository_index = successor_repository_index;
                    }
                }

                if (successor_heuristic <= (initial_heuristic_value - settings.min_improvement))
                {
                    improvement_found = true;
                }

                open_list.emplace(successor_repository_index);
            }

            if (has_solution)
            {
                const auto found_plan = state_repository.get_plan(solution_repository_index);
                output_plan.clear();
                output_plan.insert(output_plan.end(), found_plan.begin(), found_plan.end());
                break;
            }
        }

        statistics.time_end = std::chrono::high_resolution_clock::now();
        return has_solution;
    }
}  // namespace planners
