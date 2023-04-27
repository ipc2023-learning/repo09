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
#include "policy_model_actor_critic.hpp"

#include <limits>
#include <unordered_map>

namespace experiments
{
    torch::Tensor PolicyModelActorCritic::loss(uint32_t epoch, const planners::StateSpaceSampleList& batch, const PolicyBatchOutput& output)
    {
        const auto device = output.values.at(0).device();

        auto total_value_loss = torch::scalar_tensor(0.0).to(device);
        auto total_policy_loss = torch::scalar_tensor(0.0).to(device);
        auto total_goal_loss = torch::scalar_tensor(0.0).to(device);

        auto num_value_samples = 0.0;
        auto num_policy_samples = 0.0;
        auto num_goal_samples = 0.0;

        for (uint32_t sample_index = 0; sample_index < batch.size(); ++sample_index)
        {
            const auto& state = batch[sample_index].first;
            const auto& state_space = batch[sample_index].second;
            const auto& successor_states = output.state_successors[sample_index].second;

            const auto values = output.values[sample_index].view(-1);

            if (successor_states.size() == 0)
            {
                if (!state_space->is_goal_state(state))
                {
                    const auto current_value = values[0];
                    const auto target_value = 1.0 / (1.0 - get_discount_factor());
                    total_value_loss += (current_value - target_value).detach() * current_value;
                    ++num_value_samples;
                }
            }
            else
            {
                // Update value function

                const auto successor_values = values.index({ torch::indexing::Slice(1) });
                const auto successor_probabilities = output.policies[sample_index].view(-1);

                const auto optimal_value = state_space->get_distance_to_goal_state(state);
                const auto optimal_lower_bound = (1.0 - std::pow(get_discount_factor(), optimal_value)) / (1.0 - get_discount_factor());
                const auto lower_bound = disable_value_regularization_ ? 0.0 : optimal_lower_bound;
                const auto upper_bound = 1.0 / (1.0 - get_discount_factor());

                const auto cost = 1.0;
                const auto current_value = values[0];
                const auto target_value = (cost + get_discount_factor() * (successor_values * successor_probabilities).sum()).clamp(lower_bound, upper_bound);
                total_value_loss += (current_value - target_value).detach() * current_value;

                for (std::size_t successor_index = 0; successor_index < successor_states.size(); ++successor_index)
                {
                    const auto& successor_state = successor_states[successor_index];
                    const auto successor_is_goal_state = state_space->is_goal_state(successor_state);

                    if (successor_is_goal_state)
                    {
                        const auto& successor_value = successor_values[successor_index];
                        total_goal_loss += successor_value.detach() * successor_value;
                        ++num_goal_samples;
                    }
                }

                // Update policy function

                const auto step_factor = std::pow(get_discount_factor(), output.step);

                if (sample_successor_)
                {
                    const auto sampled_successor_probability = output.sampled_successor_probabilities[sample_index];
                    const auto sampled_successor_value = output.sampled_successor_values[sample_index];

                    if (disable_baseline_)
                    {
                        total_policy_loss += step_factor * get_discount_factor() * sampled_successor_value.detach() * sampled_successor_probability;
                    }
                    else
                    {
                        const auto policy_delta = ((get_discount_factor() * sampled_successor_value) - (target_value - cost)).detach();
                        total_policy_loss += step_factor * policy_delta * sampled_successor_probability;
                    }
                }
                else
                {
                    // All successors are used instead of sampling a single one

                    if (disable_baseline_)
                    {
                        total_policy_loss += (step_factor * get_discount_factor() * successor_values.detach() * successor_probabilities).sum();
                    }
                    else
                    {
                        // TODO: remove get_discount_factor()???
                        const auto policy_delta = ((get_discount_factor() * successor_values) - (target_value - cost)).detach();
                        total_policy_loss += (step_factor * policy_delta * successor_probabilities).sum();
                    }
                }

                ++num_value_samples;
                ++num_policy_samples;
            }
        }

        if (num_value_samples > 0.0)
        {
            total_value_loss /= num_value_samples;
        }

        if (num_policy_samples > 0.0)
        {
            total_policy_loss /= num_policy_samples;
        }

        if (num_goal_samples > 0.0)
        {
            total_goal_loss /= num_goal_samples;
        }

        // The learning rate for each loss should be different, normalize the scale of the gradient so that we can use the same optimizer.

        const auto VALUE_MULTIPLIER = 1.0;
        const auto POLICY_MULTIPLIER = 1.0;
        const auto GOAL_MULTIPLIER = 1.0;

        return (VALUE_MULTIPLIER * total_value_loss) + (POLICY_MULTIPLIER * total_policy_loss) + (GOAL_MULTIPLIER * total_goal_loss);
    }

    std::shared_ptr<torch::optim::Optimizer> PolicyModelActorCritic::create_optimizer(models::RelationalNeuralNetwork& model)
    {
        return std::make_shared<torch::optim::Adam>(model.parameters(), learning_rate_);
    }

    torch::Tensor PolicyModelActorCritic::train_loss(uint32_t epoch, const planners::StateSpaceSampleList& batch, const PolicyBatchOutput& output)
    {
        return this->loss(epoch, batch, output);
    }

    planners::StateSpaceSampleList
    PolicyModelActorCritic::get_batch(const std::shared_ptr<datasets::Dataset>& set, uint32_t epoch, uint32_t batch_index, uint32_t batch_size)
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
