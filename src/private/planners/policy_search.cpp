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


#include "../formalism/action.hpp"
#include "../formalism/atom.hpp"
#include "../formalism/domain.hpp"
#include "../formalism/problem.hpp"
#include "../models/relational_neural_network.hpp"
#include "generators/successor_generator_factory.hpp"
#include "policy_search.hpp"
#include "torch/torch.h"

#include <chrono>
#include <unordered_set>
#include <vector>

namespace planners
{
    PolicySearch::PolicySearch(const formalism::ProblemDescription& problem, const models::RelationalNeuralNetwork& model) :
        problem(problem),
        model(model),
        generated(0),
        expanded(0),
        time_total_ns(0),
        time_successors_ns(0),
        time_inference_ns(0)
    {
    }

    bool PolicySearch::find_plan(bool verbose, bool always_take_most_probable, bool use_closed_set, formalism::ActionList& plan)
    {
        auto current_state = problem->initial;
        auto goal_atoms = formalism::as_atoms(problem->goal);

        if (always_take_most_probable)
        {
            use_closed_set = true;
        }

        formalism::StateList state_trace({ current_state });
        formalism::ActionList action_trace;
        std::unordered_set<formalism::State> closed({ current_state });

        const auto generator = planners::create_sucessor_generator(problem, planners::SuccessorGeneratorType::AUTOMATIC);

        // Reset statistics
        generated = 0;
        expanded = 0;
        time_total_ns = 0;
        time_successors_ns = 0;
        time_inference_ns = 0;

        const auto time_total_start = std::chrono::high_resolution_clock::now();

        // Execute the policy max_horizon steps

        for (size_t step = 0; step < 10000; ++step)
        {
            if (formalism::literals_hold(problem->goal, current_state))
            {
                for (const auto& action : action_trace)
                {
                    plan.push_back(action);
                }

                break;
            }

            ++expanded;

            const auto time_successors_start = std::chrono::high_resolution_clock::now();
            formalism::StateList successors;
            formalism::ActionList actions;

            const auto ground_actions = generator->get_applicable_actions(current_state);

            for (const auto& ground_action : ground_actions)
            {
                ++generated;
                const auto successor_state = formalism::apply(ground_action, current_state);

                if (!use_closed_set || (closed.find(successor_state) == closed.end()))
                {
                    closed.insert(successor_state);
                    successors.push_back(successor_state);
                    actions.push_back(ground_action);
                }
            }

            if (successors.size() == 0)
            {
                std::cout << "No (unvisited) successors!" << std::endl;
                break;
            }

            const auto time_successors_end = std::chrono::high_resolution_clock::now();
            time_successors_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(time_successors_end - time_successors_start).count();

            // Perform a single forward pass and select successor
            const auto time_inference_start = std::chrono::high_resolution_clock::now();

            const formalism::StateTransitions state_transitions(current_state, successors, problem);
            const auto output = model.forward(state_transitions);
            const auto& policy = std::get<0>(output).view(-1);
            const auto& values = std::get<1>(output).view(-1);

            int64_t successor_index;

            if (always_take_most_probable)
            {
                successor_index = torch::argmax(policy).item<int64_t>();
            }
            else
            {
                const auto distribution_shifts = policy.cumsum(0) - torch::rand(1, policy.device());
                successor_index = std::get<0>(torch::min(distribution_shifts.clamp(0.0).nonzero(), 0, false)).item<int64_t>();
            }

            const auto selected_state = std::get<1>(state_transitions).at(successor_index);
            const auto selected_prob = policy[successor_index].item<double>();
            const auto current_value = values[0].item<double>();
            const auto successor_value = values[successor_index + 1].item<double>();

            const auto time_inference_end = std::chrono::high_resolution_clock::now();
            time_inference_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(time_inference_end - time_inference_start).count();

            if (verbose)
            {
                std::cout << "Step: " << (step + 1) << std::endl;
                std::cout << " * State: " << current_state << std::endl;
                std::cout << " * Options: ";
                for (size_t index = 0; index < actions.size(); ++index)
                {
                    const auto& action = actions[index];
                    const auto prob = policy[index].item<double>();
                    const auto value = values[index + 1].item<double>();
                    std::cout << "(" << (int32_t)(100.0 * prob + 0.5) << " %; " << std::fixed << std::setprecision(2) << value << ") " << action << "; ";
                }
                std::cout << std::endl;
            }

            if (actions.size() == 0)
            {
                std::cout << "No applicable actions!" << std::endl;
                break;
            }
            else
            {
                const auto selected_action = actions.at(successor_index);
                state_trace.push_back(selected_state);
                action_trace.push_back(selected_action);
                current_state = selected_state;

                if (verbose)
                {
                    std::cout << " * Selected: "
                              << "(" << (int32_t)(100.0 * selected_prob + 0.5) << " %; " << std::fixed << std::setprecision(2) << current_value << " -> "
                              << successor_value << ") " << selected_action << std::endl;
                }
            }
        }

        const auto time_total_end = std::chrono::high_resolution_clock::now();
        time_total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(time_total_end - time_total_start).count();

        return formalism::literals_hold(problem->goal, current_state);
    }
}  // namespace planners
