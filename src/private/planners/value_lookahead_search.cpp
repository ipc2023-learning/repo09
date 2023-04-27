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


#include "../algorithms/weisfeiler_leman.hpp"
#include "../formalism/action.hpp"
#include "../formalism/atom.hpp"
#include "../formalism/domain.hpp"
#include "../formalism/problem.hpp"
#include "../models/relational_neural_network.hpp"
#include "generators/successor_generator_factory.hpp"
#include "torch/torch.h"
#include "value_lookahead_search.hpp"

#include <chrono>
#include <queue>
#include <unordered_set>
#include <vector>

namespace std
{
    template<>
    struct hash<pair<uint64_t, uint64_t>>
    {
        size_t operator()(const pair<uint64_t, uint64_t>& x) const { return (size_t) x.first; }
    };
}  // namespace std

namespace planners
{
    struct ValueLookaheadSearch::LookaheadResult
    {
        formalism::State state;
        formalism::Action action;
        double value;
        double dead_end;

        LookaheadResult(const formalism::State& state, const formalism::Action& action, double value, double dead_end) :
            state(state),
            action(action),
            value(value),
            dead_end(dead_end)
        {
        }
    };

    std::vector<ValueLookaheadSearch::LookaheadResult> ValueLookaheadSearch::lookahead_search(const formalism::State& initial_state,
                                                                                              double initial_state_value,
                                                                                              double initial_state_dead_end,
                                                                                              double value_difference_threshold,
                                                                                              const planners::SuccessorGenerator& successor_generator)
    {
        std::unordered_map<formalism::State, formalism::State> predecessors;
        std::unordered_map<formalism::State, formalism::Action> actions;
        std::unordered_map<formalism::State, double> values;
        std::unordered_map<formalism::State, double> dead_ends;

        predecessors.insert(std::make_pair(initial_state, nullptr));
        actions.insert(std::make_pair(initial_state, nullptr));
        values.insert(std::make_pair(initial_state, initial_state_value));
        dead_ends.insert(std::make_pair(initial_state, initial_state_dead_end));

        algorithms::WeisfeilerLeman wl;
        std::unordered_set<std::pair<uint64_t, uint64_t>> wl_set;
        std::unordered_set<formalism::State> closed_set;
        formalism::StateList fringe_states;
        formalism::StateList generated_states;
        fringe_states.push_back(initial_state);

        while (true)
        {
            if (fringe_states.size() == 0)
            {
                throw std::runtime_error("fringe_states is empty, is the given state a dead end?");
            }

            const auto time_successors_start = std::chrono::high_resolution_clock::now();

            for (const auto& state : fringe_states)
            {
                const auto closed_inserted = closed_set.insert(state).second;
                auto prune_state = !closed_inserted;

                if (use_weisfeiler_leman_ && !prune_state)
                {
                    const auto state_color = wl.compute_state_color(problem_, state);
                    const auto wl_inserted = wl_set.insert(state_color).second;
                    prune_state = prune_state || !wl_inserted;
                }

                if (!prune_state)
                {
                    ++expanded;
                    const auto applicable_actions = successor_generator->get_applicable_actions(state);

                    for (const auto& action : applicable_actions)
                    {
                        ++generated;
                        const auto successor_state = formalism::apply(action, state);
                        predecessors.insert(std::make_pair(successor_state, state));
                        actions.insert(std::make_pair(successor_state, action));
                        generated_states.push_back(successor_state);
                    }
                }
            }

            const auto time_successors_end = std::chrono::high_resolution_clock::now();
            time_successors_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(time_successors_end - time_successors_start).count();

            fringe_states.clear();
            std::swap(generated_states, fringe_states);

            // Evaluate fringe states

            const auto time_inference_start = std::chrono::high_resolution_clock::now();
            const auto fringe_inference = model_.forward(fringe_states, problem_, 512);
            const auto fringe_values = fringe_inference.first.view(-1).cpu();
            const auto fringe_dead_ends = fringe_inference.second.view(-1).sigmoid().cpu();
            const auto time_inference_end = std::chrono::high_resolution_clock::now();
            time_inference_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(time_inference_end - time_inference_start).count();

            for (std::size_t index = 0; index < fringe_states.size(); ++index)
            {
                const auto& state = fringe_states[index];
                const auto state_value = fringe_values[index].item<double>();
                const auto state_dead_end = fringe_dead_ends[index].item<double>();

                values.insert(std::make_pair(state, state_value));
                dead_ends.insert(std::make_pair(state, state_dead_end));

                // TODO: Investigate how state_dead_end should be used. How to relate to initial_state_dead_end? Not make it worse by some margin?
                const auto state_satisfy_threshold = (state_dead_end < 0.5) && ((initial_state_value - state_value) > value_difference_threshold);
                const auto is_goal_state = formalism::literals_hold(problem_->goal, state);

                // TODO: If there are several states that satisfy the condition, choose the best one (except when it is a goal, then terminate immediately).

                if (state_satisfy_threshold || is_goal_state)
                {
                    std::vector<ValueLookaheadSearch::LookaheadResult> result;
                    auto backtrack_state = state;

                    while (predecessors.at(backtrack_state))
                    {
                        result.push_back(ValueLookaheadSearch::LookaheadResult(backtrack_state,
                                                                               actions.at(backtrack_state),
                                                                               values.at(backtrack_state),
                                                                               dead_ends.at(backtrack_state)));
                        backtrack_state = predecessors.at(backtrack_state);
                    }

                    std::reverse(result.begin(), result.end());
                    return result;
                }
            }
        }
    }

    ValueLookaheadSearch::ValueLookaheadSearch(const formalism::ProblemDescription& problem,
                                               const models::RelationalNeuralNetwork& model,
                                               bool use_weisfeiler_leman,
                                               uint32_t chunk_size) :
        problem_(problem),
        model_(model),
        use_weisfeiler_leman_(use_weisfeiler_leman),
        chunk_size_(chunk_size),
        generated(0),
        expanded(0),
        time_total_ns(0),
        time_successors_ns(0),
        time_inference_ns(0)
    {
    }

    bool ValueLookaheadSearch::find_plan(bool verbose,
                                         std::vector<formalism::Action>& plan_actions,
                                         std::vector<double>& plan_values,
                                         std::vector<double>& plan_dead_ends)
    {
        generated = 0;
        expanded = 0;
        time_total_ns = 0;
        time_successors_ns = 0;
        time_inference_ns = 0;

        const auto time_total_start = std::chrono::high_resolution_clock::now();

        const auto successor_generator = planners::create_sucessor_generator(problem_, planners::SuccessorGeneratorType::AUTOMATIC);
        formalism::State current_state = problem_->initial;
        double current_value;
        double current_dead_end;

        {
            const auto initial_inference = model_.forward({ problem_->initial }, problem_);
            const auto initial_value = initial_inference.first.view(-1)[0].item<double>();
            const auto initial_dead_end = initial_inference.second.view(-1).sigmoid()[0].item<double>();

            plan_values.push_back(initial_value);
            plan_dead_ends.push_back(initial_dead_end);

            current_value = initial_value;
            current_dead_end = initial_dead_end;
        }

        while (!formalism::literals_hold(problem_->goal, current_state))
        {
            const auto& lookahead = lookahead_search(current_state, current_value, current_dead_end, 2.0, successor_generator);

            for (const auto& step : lookahead)
            {
                plan_actions.push_back(step.action);
                plan_values.push_back(step.value);
                plan_dead_ends.push_back(step.dead_end);
            }

            const auto& last_step = lookahead[lookahead.size() - 1];
            current_state = last_step.state;
            current_value = last_step.value;
            current_dead_end = last_step.dead_end;
        }

        const auto time_total_end = std::chrono::high_resolution_clock::now();
        time_total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(time_total_end - time_total_start).count();

        return formalism::literals_hold(problem_->goal, current_state);
    }
}  // namespace planners
