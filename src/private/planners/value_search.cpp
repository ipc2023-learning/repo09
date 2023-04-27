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
#include "torch/torch.h"
#include "value_search.hpp"

#include <chrono>
#include <unordered_set>
#include <vector>

namespace planners
{
    ValueSearch::ValueSearch(const formalism::ProblemDescription& problem, const models::RelationalNeuralNetwork& model) :
        problem(problem),
        model(model),
        generated(0),
        expanded(0),
        time_total_ns(0),
        time_successors_ns(0),
        time_inference_ns(0)
    {
    }

    bool ValueSearch::find_plan(bool verbose, std::vector<formalism::Action>& plan)
    {
        const auto successor_generator = planners::create_sucessor_generator(problem, planners::SuccessorGeneratorType::AUTOMATIC);

        // Reset statistics
        generated = 0;
        expanded = 0;
        time_total_ns = 0;
        time_successors_ns = 0;
        time_inference_ns = 0;

        std::unordered_set<formalism::State> visited_states;
        formalism::State current_state = problem->initial;
        const auto goal_atoms = formalism::as_atoms(problem->goal);

        const auto time_total_start = std::chrono::high_resolution_clock::now();

        while (!formalism::literals_hold(problem->goal, current_state) && (plan.size() < 1000))
        {
            ++expanded;

            visited_states.insert(current_state);
            std::vector<formalism::State> successor_states;
            std::vector<formalism::Action> successor_actions;

            const auto time_successors_start = std::chrono::high_resolution_clock::now();

            for (const auto& action : successor_generator->get_applicable_actions(current_state))
            {
                const auto successor_state = formalism::apply(action, current_state);

                // if (visited_states.find(successor_state) == visited_states.end())
                {
                    ++generated;

                    successor_states.push_back(successor_state);
                    successor_actions.push_back(action);
                }
            }

            const auto time_successors_end = std::chrono::high_resolution_clock::now();
            time_successors_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(time_successors_end - time_successors_start).count();

            if (successor_states.size() > 0)
            {
                const auto time_inference_start = std::chrono::high_resolution_clock::now();

                const auto outputs = model.forward(successor_states, problem);
                const auto values = outputs.first.view(-1);
                const auto dead_ends = outputs.second.sigmoid().view(-1);
                const auto min_index = (uint32_t) torch::argmin(1000.0 * dead_ends.round() + values).item<double>();

                const auto time_inference_end = std::chrono::high_resolution_clock::now();
                time_inference_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(time_inference_end - time_inference_start).count();

                if (verbose)
                {
                    std::cout << "Step: " << (plan.size() + 1) << std::endl;
                    std::cout << " * State: " << current_state << std::endl;
                    std::cout << " * Options: ";

                    for (size_t index = 0; index < successor_states.size(); ++index)
                    {
                        const auto& action = successor_actions[index];
                        const auto value = values[index].item<double>();
                        const auto dead_end = dead_ends[index].item<double>();

                        std::cout << "(" << std::fixed << std::setprecision(2) << value << ", " << dead_end << ") " << action << "; ";
                    }

                    std::cout << std::endl
                              << " * Selected: "
                              << "(" << std::fixed << std::setprecision(2) << values[min_index].item<double>() << ") " << successor_actions[min_index]
                              << std::endl
                              << std::endl;
                }

                current_state = successor_states.at(min_index);
                plan.push_back(successor_actions.at(min_index));
            }
            else
            {
                std::cout << "No unvisited successor states!" << std::endl;
                break;
            }
        }

        const auto time_total_end = std::chrono::high_resolution_clock::now();
        time_total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(time_total_end - time_total_start).count();

        return formalism::literals_hold(problem->goal, current_state);
    }
}  // namespace planners
