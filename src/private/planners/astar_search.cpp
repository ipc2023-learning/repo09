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


// #include "astar_search.hpp"

// #include <algorithm>
// #include <chrono>
// #include <cmath>
// #include <deque>
// #include <limits>
// #include <queue>
// #include <unordered_map>

// namespace planners
// {
//     AStarSearch::AStarSearch() : generated(0), expanded(0), max_expanded(std::numeric_limits<uint32_t>::max()), time_total_ns(0), time_successors_ns(0) {}

//     bool AStarSearch::find_plan(const formalism::ProblemDescription& problem,
//                                 planners::SuccessorGenerator& successor_generator,
//                                 planners::HeuristicBase& heuristic,
//                                 planners::ConditionBase& goal,
//                                 planners::ConditionBase& prune,
//                                 std::vector<formalism::Action>& out_plan)
//     {
//         generated = 0;
//         expanded = 0;
//         time_total_ns = 0;
//         time_successors_ns = 0;

//         struct QueueNode
//         {
//             uint32_t state_index;
//             int32_t g_value;
//             int32_t h_value;

//             bool operator<(const QueueNode& other) const { return (g_value + h_value) > (other.g_value + other.h_value); }
//         };

//         std::priority_queue<QueueNode> queue;

//         {
//             uint32_t initial_index;
//             repository.add_or_get_state(problem->initial, nullptr, std::numeric_limits<uint32_t>::max(), initial_index);
//             auto& initial_context = repository.get_context(initial_index);

//             const auto initial_heuristic = heuristic.get_cost(problem->initial);
//             initial_context.is_goal_state = goal.test(problem->initial, initial_heuristic);

//             queue.push({ initial_index, 0, initial_heuristic });
//         }

//         const auto time_start = std::chrono::high_resolution_clock::now();
//         auto found_solution = false;

//         while (!queue.empty() && (expanded < max_expanded))
//         {
//             // TODO: Get all of the nodes with the same g + h value as the top node.
//             // TODO: Expand all of the collected nodes with the successor_generator and create a batch with all the successors.
//             // TODO: Compute the heuristic value for all successor states with costs=heuristic.get_cost(batch).
//             // TODO: Prune away successor states with prune.test(batch, heuristics).
//             // TODO: Add the remaining successor states to repository and queue.

//             auto& state_node = queue.top();
//             auto& state_index = state_node.state_index;
//             auto& state_context = repository.get_context(state_index);
//             queue.pop();

//             if (state_context.is_goal_state)
//             {
//                 const auto found_plan = repository.get_plan(state_index);
//                 found_solution = true;

//                 out_plan.clear();
//                 out_plan.insert(out_plan.end(), found_plan.begin(), found_plan.end());
//                 break;
//             }

//             if (state_context.is_expanded)
//             {
//                 continue;
//             }

//             state_context.is_expanded = true;
//             ++expanded;

//             const auto grounding_time_start = std::chrono::high_resolution_clock::now();
//             const auto state = repository.get_state(state_index);
//             const auto applicable_actions = successor_generator->get_applicable_actions(state);
//             const auto grounding_time_end = std::chrono::high_resolution_clock::now();
//             time_successors_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(grounding_time_end - grounding_time_start).count();

//             for (const auto& action : applicable_actions)
//             {
//                 ++generated;
//                 uint32_t successor_state_index;
//                 const auto successor_state = formalism::apply(action, state);

//                 if (repository.add_or_get_state(successor_state, action, state_index, successor_state_index))
//                 {
//                     auto& successor_context = repository.get_context(successor_state_index);
//                     const auto successor_heuristic = heuristic.get_cost(successor_state);
//                     successor_context.is_goal_state = goal.test(successor_state, successor_heuristic);
//                     queue.push({ successor_state_index, state_node.g_value + action->cost, successor_heuristic });
//                 }
//                 // TODO: If add_or_get_state returns false, the successor_state_index points to an already existing entry.
//                 // Check if the state is expanded, if not, check if the new g + h cost is less than the current g + h cost, if so, reprioritize the state
//                 with
//                 // the new cost.
//             }
//         }

//         const auto time_end = std::chrono::high_resolution_clock::now();
//         time_total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_start).count();

//         return found_solution;
//     }

//     void AStarSearch::set_max_expanded(uint32_t max) { max_expanded = max; }
// }  // namespace planners
