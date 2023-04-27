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
#include "policy_iterative_tabular_probability.hpp"

#include <limits>
#include <unordered_map>

namespace experiments
{
    PolicyIterativeTabularProbability::PolicyIterativeTabularProbability(uint32_t batch_size,
                                                                         uint32_t chunk_size,
                                                                         uint32_t max_epochs,
                                                                         double learning_rate,
                                                                         double discount_factor,
                                                                         bool disable_balancing,
                                                                         bool disable_baseline,
                                                                         bool disable_value_regularization) :
        PolicyDatasetExperiment(batch_size, chunk_size, max_epochs, 1, discount_factor, disable_balancing, true, true, PolicySamplingMethod::Policy),
        learning_rate_(learning_rate),
        disable_baseline_(disable_baseline),
        disable_value_regularization_(disable_value_regularization),
        problem_optimal_vector_(),
        problem_value_vector_(),
        problem_probability_vector_() {};

    void PolicyIterativeTabularProbability::initialize_tables_if_necessary(const planners::StateSpace& state_space, models::RelationalNeuralNetwork& model)
    {
        if (problem_value_vector_.find(state_space->problem) == problem_value_vector_.end())
        {
            auto transition_probabilities = create_transition_probabilities(state_space);
            auto batches_indices = create_batches(state_space->problem, transition_probabilities, get_batch_size());
            update_transition_probabilities(batches_indices.first, batches_indices.second, transition_probabilities, model);
            const auto values = compute_policy_evaluation(transition_probabilities, get_discount_factor());
            const auto probabilities = compute_probability_evaluation(transition_probabilities, values);
            problem_value_vector_.insert(std::make_pair(state_space->problem, std::move(values)));
            problem_probability_vector_.insert(std::make_pair(state_space->problem, std::move(probabilities)));
            problem_optimal_vector_.insert(std::make_pair(state_space->problem, compute_optimal_policy_evaluation(state_space, get_discount_factor())));
        }
    }

    torch::Tensor PolicyIterativeTabularProbability::loss(const planners::StateSpaceSampleList& batch,
                                                          const PolicyBatchOutput& output,
                                                          models::RelationalNeuralNetwork& model)
    {
        const auto device = output.policies.at(0).device();
        auto total_value_loss = torch::scalar_tensor(0.0).to(device);
        auto total_policy_loss = torch::scalar_tensor(0.0).to(device);

        for (uint32_t sample_index = 0; sample_index < batch.size(); ++sample_index)
        {
            const auto& state = batch[sample_index].first;
            const auto& state_space = batch[sample_index].second;
            const auto& value_output = output.values[sample_index].view(-1);
            const auto& policy_output = output.policies[sample_index].view(-1);
            const auto& sampled_successor_probability = output.sampled_successor_probabilities[sample_index];
            const auto& sampled_successor_state = output.sampled_successor_states[sample_index];
            const auto& successor_states = output.state_successors[sample_index].second;

            if (sampled_successor_state == nullptr)
            {
                continue;
            }

            initialize_tables_if_necessary(state_space, model);
            auto& value_vector = problem_value_vector_.at(state_space->problem);
            auto& descending_vector = problem_probability_vector_.at(state_space->problem);
            auto& optimal_vector = problem_optimal_vector_.at(state_space->problem);

            const auto transition_probabilities = policy_output.detach().cpu();
            const auto state_index = state_space->get_unique_index_of_state(state);

            // Compute and update the value of the state, and the probability of going to a successor with a lower value.

            const auto reward = 1.0;
            double state_value = reward;
            double state_probability = 0.0;

            for (std::size_t transition_index = 0; transition_index < successor_states.size(); ++transition_index)
            {
                const auto transition_probability = transition_probabilities[transition_index].item<double>();
                const auto successor_state = successor_states[transition_index];
                const auto successor_index = state_space->get_unique_index_of_state(successor_state);
                const auto successor_value = value_vector[successor_index];
                const auto trajectory_probability = descending_vector[successor_index];
                state_value += get_discount_factor() * transition_probability * successor_value;

                if ((successor_value + (0.9 * get_discount_factor())) < value_vector[state_index])
                {
                    state_probability += transition_probability * trajectory_probability;
                }

                // Update value if successor is a goal state

                if (!disable_value_regularization_ && state_space->is_goal_state(successor_state))
                {
                    total_value_loss += value_output[transition_index + 1].abs();
                }
            }

            value_vector[state_index] = state_value;
            descending_vector[state_index] = state_probability;

            if (!disable_value_regularization_)
            {
                const auto value_optimal = optimal_vector[state_index];
                total_value_loss += (value_optimal - value_output[0]).abs();
            }

            // Update policy

            const auto samled_successor_index = state_space->get_unique_index_of_state(sampled_successor_state);
            const auto sampled_successor_descending = descending_vector[samled_successor_index];
            const auto state_baseline = disable_baseline_ ? 0.0 : (1.0 - descending_vector[state_index]);
            total_policy_loss += ((1.0 - sampled_successor_descending) - state_baseline) * sampled_successor_probability.log();
        }

        if (batch.size() > 0)
        {
            total_value_loss /= (double) batch.size();
            total_policy_loss /= (double) batch.size();
        }

        return total_policy_loss + total_value_loss;
    }

    std::shared_ptr<torch::optim::Optimizer> PolicyIterativeTabularProbability::create_optimizer(models::RelationalNeuralNetwork& model)
    {
        return std::make_shared<torch::optim::Adam>(model.parameters(), learning_rate_);
    }

    torch::Tensor PolicyIterativeTabularProbability::train_loss(uint32_t epoch,
                                                                const planners::StateSpaceSampleList& batch,
                                                                const PolicyBatchOutput& output,
                                                                models::RelationalNeuralNetwork& model)
    {
        return this->loss(batch, output, model);
    }

    planners::StateSpaceSampleList
    PolicyIterativeTabularProbability::get_batch(const std::shared_ptr<datasets::Dataset>& set, uint32_t epoch, uint32_t batch_index, uint32_t batch_size)
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
