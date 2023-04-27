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


#include "../models/utils.hpp"
#include "policy_evaluation_utils.hpp"

namespace experiments
{
    TransitionProbabilities create_transition_probabilities(const planners::StateSpace& state_space)
    {
        // Initialize policy evaluation for problem.
        TransitionProbabilities policy_evaluation;
        const auto n = state_space->num_states();
        policy_evaluation.states.resize(n);
        policy_evaluation.is_goal_state.resize(n);
        policy_evaluation.is_dead_end_state.resize(n);
        policy_evaluation.neighbor_indices.resize(n);
        policy_evaluation.probabilities.resize(n);

        for (const auto& state : state_space->get_states())
        {
            const auto state_index = state_space->get_unique_index_of_state(state);
            policy_evaluation.states[state_index] = state;
            policy_evaluation.is_goal_state[state_index] = state_space->is_goal_state(state);
            policy_evaluation.is_dead_end_state[state_index] = state_space->is_dead_end_state(state);
        }

        for (std::size_t state_index = 0; state_index < policy_evaluation.states.size(); ++state_index)
        {
            if (!policy_evaluation.is_dead_end_state[state_index] && !policy_evaluation.is_goal_state[state_index])
            {
                auto& transition_probabilities = policy_evaluation.probabilities[state_index];
                auto& neighbor_indices = policy_evaluation.neighbor_indices[state_index];

                const auto& state = policy_evaluation.states[state_index];
                const auto transitions = state_space->get_forward_transitions(state);

                for (const auto& transition : transitions)
                {
                    const auto& neighbor_state = transition->target_state;
                    neighbor_indices.push_back(state_space->get_unique_index_of_state(neighbor_state));
                }

                transition_probabilities.resize(neighbor_indices.size());
                const auto transition_probability = 1.0 / (double) neighbor_indices.size();
                std::fill(transition_probabilities.begin(), transition_probabilities.end(), transition_probability);
            }
        }

        return policy_evaluation;
    }

    std::pair<std::vector<formalism::StateTransitionsVector>, std::vector<std::vector<std::size_t>>>
    create_batches(const formalism::ProblemDescription& problem, const TransitionProbabilities& transition_probabilities, const uint32_t batch_size)
    {
        formalism::StateTransitionsVector transitions;
        std::vector<std::size_t> state_indices;

        for (std::size_t state_index = 0; state_index < transition_probabilities.states.size(); ++state_index)
        {
            if (!transition_probabilities.is_dead_end_state[state_index] && !transition_probabilities.is_goal_state[state_index])
            {
                const auto& state = transition_probabilities.states[state_index];
                std::vector<formalism::State> successors;

                auto& neighbor_indices = transition_probabilities.neighbor_indices[state_index];

                for (std::size_t transition_index = 0; transition_index < neighbor_indices.size(); ++transition_index)
                {
                    const auto neighbor_index = neighbor_indices[transition_index];
                    const auto& neighbor_state = transition_probabilities.states[neighbor_index];
                    successors.push_back(neighbor_state);
                }

                transitions.push_back(formalism::StateTransitions(state, std::move(successors), problem));
                state_indices.push_back(state_index);
            }
        }

        std::vector<formalism::StateTransitionsVector> batch_transitions;
        std::vector<std::vector<std::size_t>> batch_state_indices;

        formalism::StateTransitionsVector current_batch;
        std::vector<std::size_t> current_indices;
        std::size_t current_batch_size = 0;

        for (std::size_t transition_index = 0; transition_index < transitions.size(); ++transition_index)
        {
            const auto& transition = transitions[transition_index];
            const auto state_index = state_indices[transition_index];
            const auto num_successors = std::get<1>(transition).size();

            if ((current_batch_size > 0) && ((current_batch_size + num_successors + 1) > batch_size))
            {
                batch_transitions.push_back(current_batch);
                batch_state_indices.push_back(current_indices);
                current_batch.clear();
                current_indices.clear();
                current_batch_size = 0;
            }

            current_batch.push_back(transition);
            current_indices.push_back(state_index);
            current_batch_size += num_successors + 1;
        }

        if (current_batch_size > 0)
        {
            batch_transitions.push_back(current_batch);
            batch_state_indices.push_back(current_indices);
            current_batch.clear();
            current_indices.clear();
            current_batch_size = 0;
        }

        return std::make_pair(batch_transitions, batch_state_indices);
    }

    void update_transition_probabilities(const std::vector<formalism::StateTransitionsVector>& batches,
                                         const std::vector<std::vector<std::size_t>>& state_index_list,
                                         TransitionProbabilities& transition_probabilities,
                                         models::RelationalNeuralNetwork& model)
    {
        torch::NoGradGuard no_grad;

        for (std::size_t batch_index = 0; batch_index < batches.size(); ++batch_index)
        {
            const auto& batch = batches[batch_index];
            const auto& state_indices = state_index_list[batch_index];
            const auto output = std::get<0>(model.forward(batch));

            for (std::size_t output_index = 0; output_index < output.size(); ++output_index)
            {
                const auto state_index = state_indices[output_index];

                // TODO: Find a better way to do this.
                for (std::size_t i = 0; i < transition_probabilities.neighbor_indices[state_index].size(); ++i)
                {
                    transition_probabilities.probabilities[state_index][i] = output[output_index][i].item<double>();
                }
            }
        }
    }

    std::vector<double> compute_optimal_policy_evaluation(const planners::StateSpace& state_space, const double discount_factor)
    {
        std::vector<double> values;
        values.resize(state_space->num_states());

        for (const auto& state : state_space->get_states())
        {
            const auto state_index = state_space->get_unique_index_of_state(state);

            if (state_space->is_dead_end_state(state))
            {
                values[state_index] = 1.0 / (1.0 - discount_factor);
            }
            else
            {
                const auto state_value = state_space->get_distance_to_goal_state(state);
                values[state_index] = (std::pow(discount_factor, state_value) - 1.0) / (discount_factor - 1.0);
            }
        }

        return values;
    }

    std::vector<double> compute_policy_evaluation(const TransitionProbabilities& transition_probabilities, const double discount_factor)
    {
        std::vector<double> values(transition_probabilities.states.size());

        while (true)
        {
            double max_update_delta = 0.0;

            for (std::size_t state_index = 0; state_index < transition_probabilities.states.size(); ++state_index)
            {
                if (transition_probabilities.is_goal_state[state_index])
                {
                    values[state_index] = 0.0;
                }
                else if (transition_probabilities.is_dead_end_state[state_index])
                {
                    values[state_index] = 1.0 / (1.0 - discount_factor);
                }
                else
                {
                    double probability_sum = 0.0;
                    const auto& neighbor_indices = transition_probabilities.neighbor_indices[state_index];
                    const auto& probabilities = transition_probabilities.probabilities[state_index];

                    for (std::size_t transition_index = 0; transition_index < probabilities.size(); ++transition_index)
                    {
                        probability_sum += probabilities[transition_index] * values[neighbor_indices[transition_index]];
                    }

                    const auto updated_value = 1.0 + discount_factor * probability_sum;
                    max_update_delta = std::max(max_update_delta, std::abs(values[state_index] - updated_value));
                    values[state_index] = updated_value;
                }
            }

            if (max_update_delta < 0.01)
            {
                break;
            }
        }

        return values;
    }

    std::vector<double> compute_probability_evaluation(const TransitionProbabilities& transition_probabilities, const std::vector<double>& values)
    {
        std::vector<double> probs(transition_probabilities.states.size());

        while (true)
        {
            double max_update_delta = 0.0;

            for (std::size_t state_index = 0; state_index < transition_probabilities.states.size(); ++state_index)
            {
                if (transition_probabilities.is_goal_state[state_index])
                {
                    max_update_delta = std::max(max_update_delta, 1.0 - probs[state_index]);
                    probs[state_index] = 1.0;
                }
                else if (transition_probabilities.is_dead_end_state[state_index])
                {
                    probs[state_index] = 0.0;
                }
                else
                {
                    double probability_sum = 0.0;
                    const auto& neighbor_indices = transition_probabilities.neighbor_indices[state_index];
                    const auto& probabilities = transition_probabilities.probabilities[state_index];

                    for (std::size_t transition_index = 0; transition_index < probabilities.size(); ++transition_index)
                    {
                        if ((values[neighbor_indices[transition_index]] + 0.75) < values[state_index])
                        {
                            probability_sum += probabilities[transition_index] * probs[neighbor_indices[transition_index]];
                        }
                    }

                    const auto updated_value = probability_sum;
                    max_update_delta = std::max(max_update_delta, std::abs(probs[state_index] - updated_value));
                    probs[state_index] = updated_value;
                }
            }

            if (max_update_delta < 0.01)
            {
                break;
            }
        }

        return probs;
    }
}  // namespace experiments
