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


#include "bfs_search.hpp"

#include <algorithm>
#include <chrono>
#include <deque>
#include <limits>
#include <unordered_map>

namespace planners
{
    struct StateContext
    {
        formalism::Action predecessor_action;
        uint32_t predecessor_state_index;
        uint32_t fringe_value;
        bool is_goal_state;
        bool is_expanded;

        StateContext(const formalism::Action& predecessor_action,
                     const uint32_t predecessor_state_index,
                     const uint32_t fringe_value,
                     const bool is_goal_state,
                     const bool is_expanded) :
            predecessor_action(predecessor_action),
            predecessor_state_index(predecessor_state_index),
            fringe_value(fringe_value),
            is_goal_state(is_goal_state),
            is_expanded(is_expanded)
        {
        }
    };

    bool add_or_get_state(const formalism::State& state,
                          const formalism::LiteralList goal,
                          const formalism::Action& predecessor_action,
                          const uint32_t predecessor_state_index,
                          const uint32_t fringe_value,
                          std::vector<formalism::State>& states,
                          std::vector<StateContext>& contexts,
                          std::unordered_map<formalism::State, uint32_t>& indices,
                          uint32_t& out_index)
    {
        const auto index_handler = indices.find(state);  // TODO: Profiling 20 %

        if (index_handler != indices.end())
        {
            out_index = index_handler->second;
            return false;
        }
        else
        {
            out_index = contexts.size();
            const auto is_goal_state = literals_hold(goal, state);
            states.push_back(state);
            contexts.push_back(StateContext(predecessor_action, predecessor_state_index, fringe_value, is_goal_state, false));
            indices.insert(std::make_pair(state, out_index));
            return true;
        }
    }

    BreadthFirstSearch::BreadthFirstSearch(const formalism::ProblemDescription& problem, planners::SuccessorGeneratorType successor_generator_type) :
        problem_(problem),
        successor_generator_type_(successor_generator_type),
        print_(false),
        generated(0),
        expanded(0),
        max_expanded(std::numeric_limits<uint32_t>::max()),
        time_total_ns(0),
        time_successors_ns(0)
    {
    }

    bool BreadthFirstSearch::find_plan(std::vector<formalism::Action>& plan)
    {
        generated = 0;
        expanded = 0;
        time_total_ns = 0;
        time_successors_ns = 0;

        const auto successor_generator = planners::create_sucessor_generator(problem_, successor_generator_type_);

        const auto goal = problem_->goal;
        const auto initial_state = problem_->initial;

        std::vector<formalism::State> states;
        std::vector<StateContext> contexts;
        std::unordered_map<formalism::State, uint32_t> indices;
        std::deque<uint32_t> queue;

        {
            uint32_t initial_index;
            add_or_get_state(initial_state, goal, nullptr, std::numeric_limits<uint32_t>::max(), 0, states, contexts, indices, initial_index);
            queue.push_back(initial_index);
        }

        uint32_t last_fringe_value = 0;

        const auto time_start = std::chrono::high_resolution_clock::now();
        auto found_solution = false;

        while ((queue.size() > 0) && (expanded < max_expanded))
        {
            const auto state_index = queue.front();
            queue.pop_front();
            auto state_context = contexts[state_index];

            if (state_context.is_expanded)
            {
                continue;
            }

            if (print_ && (state_context.fringe_value > last_fringe_value))
            {
                last_fringe_value = state_context.fringe_value;
                const auto fringe_time_end = std::chrono::high_resolution_clock::now();
                const auto fringe_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(fringe_time_end - time_start).count();
                std::cout << "[f = " << state_context.fringe_value << "] Expanded: " << expanded << "; Generated: " << generated << " (" << fringe_time_ms
                          << " ms)" << std::endl;
            }

            if (state_context.is_goal_state)
            {
                plan.clear();
                auto current_state_index = state_index;

                while (true)
                {
                    const auto& current_context = contexts[current_state_index];
                    const auto predecessor_action = current_context.predecessor_action;

                    if (predecessor_action == nullptr)
                    {
                        break;
                    }

                    plan.push_back(predecessor_action);
                    current_state_index = current_context.predecessor_state_index;
                }

                std::reverse(plan.begin(), plan.end());

                found_solution = true;
                break;
            }

            state_context.is_expanded = true;
            contexts[state_index] = state_context;
            ++expanded;

            const auto grounding_time_start = std::chrono::high_resolution_clock::now();
            const auto state = states[state_index];
            const auto applicable_actions = successor_generator->get_applicable_actions(state);
            const auto grounding_time_end = std::chrono::high_resolution_clock::now();
            time_successors_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(grounding_time_end - grounding_time_start).count();

            for (const auto& action : applicable_actions)
            {
                ++generated;
                uint32_t successor_state_index;
                const auto successor_state = formalism::apply(action, state);

                if (add_or_get_state(successor_state,
                                     goal,
                                     action,
                                     state_index,
                                     state_context.fringe_value + 1,
                                     states,
                                     contexts,
                                     indices,
                                     successor_state_index))
                {
                    queue.push_back(successor_state_index);
                }
            }
        }

        const auto time_end = std::chrono::high_resolution_clock::now();
        time_total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_start).count();
        return found_solution;
    }

    void BreadthFirstSearch::print_progress(bool flag) { print_ = flag; }

    void BreadthFirstSearch::set_max_expanded(uint32_t max) { max_expanded = max; }
}  // namespace planners
