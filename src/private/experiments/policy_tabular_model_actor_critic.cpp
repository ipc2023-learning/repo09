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
#include "policy_tabular_model_actor_critic.hpp"

#include <limits>
#include <unordered_map>

namespace experiments
{
    PolicyTabularModelActorCritic::PolicyTabularModelActorCritic(uint32_t batch_size,
                                                                 uint32_t chunk_size,
                                                                 uint32_t max_epochs,
                                                                 uint32_t trajectory_length,
                                                                 double learning_rate,
                                                                 double discount_factor,
                                                                 bool disable_balancing,
                                                                 bool disable_baseline) :
        PolicyDatasetExperiment(batch_size,
                                chunk_size,
                                max_epochs,
                                trajectory_length,
                                discount_factor,
                                disable_balancing,
                                true,
                                true,
                                PolicySamplingMethod::Uniform),
        learning_rate_(learning_rate),
        disable_baseline_(disable_baseline),
        problem_value_vector_() {};

    void PolicyTabularModelActorCritic::initialize_probabilities_and_values(const planners::StateSpaceSampleList& batch, models::RelationalNeuralNetwork& model)
    {
        torch::NoGradGuard no_grad;
        model.eval();

        const auto discount_factor = get_discount_factor();

        for (const auto& sample : batch)
        {
            const auto& state_space = sample.second;

            if (problem_value_vector_.find(state_space->problem) == problem_value_vector_.end())
            {
                std::vector<double> value_vector(state_space->num_states(), 0.0);
                std::vector<std::vector<std::pair<uint64_t, double>>> transitions_vector(state_space->num_states(), std::vector<std::pair<uint64_t, double>>());

                for (const auto& state : state_space->get_states())
                {
                    const auto state_index = state_space->get_unique_index_of_state(state);

                    if (state_space->is_dead_end_state(state))
                    {
                        value_vector[state_index] = 1.0 / (1.0 - discount_factor);
                    }
                    else if (state_space->is_goal_state(state))
                    {
                        value_vector[state_index] = 0.0;
                    }
                    else
                    {
                        value_vector[state_index] = 0.5 * (1.0 / (1.0 - discount_factor));

                        // TODO: Optimize this. I think we can forward(const formalism::StateTransitionsVector& state_transitions) to batch things better.
                        const auto& transitions = state_space->get_forward_transitions(state);
                        // const auto output = model.forward(formalism::to_state_transitions(state_space->problem, transitions));
                        // const auto successor_probabilities = std::get<0>(output);
                        auto& transition_values = transitions_vector[state_index];

                        for (std::size_t transition_index = 0; transition_index < transitions.size(); ++transition_index)
                        {
                            const auto successor_index = state_space->get_unique_index_of_state(transitions[transition_index]->target_state);
                            // const auto successor_probability = successor_probabilities[transition_index].item<double>();
                            const auto successor_probability = 1.0 / (double) transitions.size();
                            transition_values.push_back(std::make_pair(successor_index, successor_probability));
                        }
                    }
                }

                problem_value_vector_.insert(std::make_pair(state_space->problem, std::move(value_vector)));
                problem_transitions_vector_.insert(std::make_pair(state_space->problem, std::move(transitions_vector)));
            }
        }
    }
    void PolicyTabularModelActorCritic::update_transition_probabilities(const planners::StateSpaceSampleList& batch, const PolicyBatchOutput& output)
    {
        // torch::NoGradGuard no_grad;

        for (std::size_t sample_index = 0; sample_index < batch.size(); ++sample_index)
        {
            const auto& state = batch[sample_index].first;
            const auto& state_space = batch[sample_index].second;

            if (!state_space->is_dead_end_state(state) && !state_space->is_goal_state(state))
            {
                const auto& successor_states = output.state_successors[sample_index].second;
                const auto successor_probabilities = output.policies[sample_index].view(-1);

                const auto state_index = state_space->get_unique_index_of_state(state);
                auto& state_transitions = problem_transitions_vector_.at(state_space->problem)[state_index];

                if (state_transitions.size() != successor_states.size())
                {
                    throw std::runtime_error("sizes of state_transitions and successor_states do not match");
                }

                for (std::size_t transition_index = 0; transition_index < successor_states.size(); ++transition_index)
                {
                    const auto& successor_state = successor_states[transition_index];
                    const auto successor_index = state_space->get_unique_index_of_state(successor_state);
                    const auto successor_probability = successor_probabilities[transition_index].item<double>();
                    state_transitions[transition_index] = std::make_pair(successor_index, successor_probability);
                }
            }
        }
    }

    void PolicyTabularModelActorCritic::update_values(const planners::StateSpaceSampleList& batch)
    {
        const auto discount_factor = get_discount_factor();

        for (const auto& sample : batch)
        {
            const auto& problem = sample.second->problem;
            const auto& transitions_vector = problem_transitions_vector_.at(problem);
            auto& value_vector = problem_value_vector_.at(problem);

            while (true)
            {
                double max_update_delta = 0.0;

                for (std::size_t state_index = 0; state_index < value_vector.size(); ++state_index)
                {
                    const auto& transitions = transitions_vector[state_index];

                    if (transitions.size() > 0)
                    {
                        // By design, dead-end states and goal states have no successors, and we do not need to update their value.

                        double probability_sum = 0.0;

                        for (const auto& transition : transitions)
                        {
                            const auto target_index = transition.first;
                            const auto target_probability = transition.second;
                            probability_sum += target_probability * value_vector[target_index];
                        }

                        const auto updated_value = 1.0 + discount_factor * probability_sum;
                        max_update_delta = std::max(max_update_delta, std::abs(value_vector[state_index] - updated_value));
                        value_vector[state_index] = updated_value;
                    }
                }

                if (max_update_delta < 0.01)
                {
                    break;
                }
            }
        }
    }

    torch::Tensor
    PolicyTabularModelActorCritic::loss(const planners::StateSpaceSampleList& batch, const PolicyBatchOutput& output, models::RelationalNeuralNetwork& model)
    {
        initialize_probabilities_and_values(batch, model);
        update_transition_probabilities(batch, output);
        update_values(batch);

        const auto device = output.policies.at(0).device();
        auto total_policy_loss = torch::scalar_tensor(0.0, { device });
        auto num_policy_samples = 0.0;

        for (std::size_t sample_index = 0; sample_index < batch.size(); ++sample_index)
        {
            const auto& state = batch[sample_index].first;
            const auto& state_space = batch[sample_index].second;

            if (!state_space->is_dead_end_state(state) && !state_space->is_goal_state(state))
            {
                const auto& successor_states = output.state_successors[sample_index].second;
                const auto successor_probabilities = output.policies[sample_index].view(-1);

                const auto cost = 1.0;
                const auto& value_vector = problem_value_vector_.at(state_space->problem);
                const auto target_value = disable_baseline_ ? cost : value_vector[state_space->get_unique_index_of_state(state)];

                const auto discount_factor = get_discount_factor();
                const auto step_factor = std::pow(discount_factor, output.step);

                for (std::size_t transition_index = 0; transition_index < successor_states.size(); ++transition_index)
                {
                    const auto& successor_state = successor_states[transition_index];
                    const auto successor_index = state_space->get_unique_index_of_state(successor_state);
                    const auto successor_value = value_vector[successor_index];
                    const auto successor_probability = successor_probabilities[transition_index];

                    const auto delta = (discount_factor * successor_value) - (target_value - cost);
                    total_policy_loss += step_factor * delta * successor_probability;
                }

                ++num_policy_samples;
            }
        }

        if (num_policy_samples > 0.0)
        {
            total_policy_loss /= num_policy_samples;
        }

        return total_policy_loss;
    }

    std::shared_ptr<torch::optim::Optimizer> PolicyTabularModelActorCritic::create_optimizer(models::RelationalNeuralNetwork& model)
    {
        return std::make_shared<torch::optim::Adam>(model.parameters(), learning_rate_);
    }

    torch::Tensor PolicyTabularModelActorCritic::train_loss(uint32_t epoch,
                                                            const planners::StateSpaceSampleList& batch,
                                                            const PolicyBatchOutput& output,
                                                            models::RelationalNeuralNetwork& model)
    {
        return this->loss(batch, output, model);
    }

    planners::StateSpaceSampleList
    PolicyTabularModelActorCritic::get_batch(const std::shared_ptr<datasets::Dataset>& set, uint32_t epoch, uint32_t batch_index, uint32_t batch_size)
    {
        planners::StateSpaceSampleList batch;
        uint32_t max_index_excl = std::min((uint32_t) set->size(), batch_index * batch_size + batch_size);

        for (uint32_t index = (batch_index * batch_size); index < max_index_excl; ++index)
        {
            const auto sample = set->get(index);
            const auto& state = sample.first;
            const auto& state_space = sample.second;

            if (state_space->is_dead_end_state(state))
            {
                throw std::invalid_argument("set contains dead-end states");
            }

            if (state_space->is_goal_state(state))
            {
                throw std::invalid_argument("set contains goal states");
            }

            batch.push_back(sample);
        }

        return batch;
    }
}  // namespace experiments
