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
#include "policy_suboptimal_model_actor_critic.hpp"

#include <limits>
#include <unordered_map>

namespace experiments
{
    torch::Tensor PolicyIterativeFullProbability::loss(uint32_t epoch, const planners::StateSpaceSampleList& batch, const PolicyBatchOutput& output)
    {
        const auto device = output.values.at(0).device();

        auto total_value_loss = torch::scalar_tensor(0.0, { device });
        auto total_policy_loss = torch::scalar_tensor(0.0, { device });
        auto total_descending_loss = torch::scalar_tensor(0.0, { device });
        auto total_goal_value_loss = torch::scalar_tensor(0.0, { device });
        auto total_goal_descending_loss = torch::scalar_tensor(0.0, { device });

        auto num_value_samples = 0.0;
        auto num_policy_samples = 0.0;
        auto num_descending_samples = 0.0;
        auto num_goal_value_samples = 0.0;
        auto num_goal_descending_samples = 0.0;

        for (uint32_t sample_index = 0; sample_index < batch.size(); ++sample_index)
        {
            const auto& state = batch[sample_index].first;
            const auto& state_space = batch[sample_index].second;
            const auto& successor_states = output.state_successors[sample_index].second;

            const auto values = output.values[sample_index].view(-1);
            const auto discount_factor = get_discount_factor();

            if (successor_states.size() == 0)
            {
                if (!state_space->is_goal_state(state))
                {
                    const auto current_value = values[0];
                    const auto target_value = 1.0 / (1.0 - discount_factor);
                    total_value_loss += (current_value - target_value).detach() * current_value;
                    ++num_value_samples;
                }
            }
            else
            {
                const auto descending_logits = -output.dead_ends[sample_index].view(-1);  // Flip the sign of the logit to get 1 - x.
                const auto successor_values = values.index({ torch::indexing::Slice(1) });
                const auto successor_descending_logits = descending_logits.index({ torch::indexing::Slice(1) });
                const auto successor_descending_probs = successor_descending_logits.sigmoid();
                const auto successor_transition_probs = output.policies[sample_index].view(-1);

                const auto optimal_value = state_space->get_distance_to_goal_state(state);
                const auto optimal_lower_bound = (1.0 - std::pow(discount_factor, optimal_value)) / (1.0 - discount_factor);
                const auto lower_bound = disable_value_regularization_ ? 0.0 : optimal_lower_bound;
                const auto upper_bound = 1.0 / (1.0 - discount_factor);

                const auto cost = 1.0;
                const auto current_value = values[0];
                const auto target_value = (cost + discount_factor * (successor_values * successor_transition_probs).sum()).clamp(lower_bound, upper_bound);
                total_value_loss += (current_value - target_value).detach() * current_value;

                const auto descending_mask = (successor_values + 0.75) < current_value;
                const auto descending_probability = (descending_mask * (successor_transition_probs * successor_descending_probs)).sum();
                const auto current_descending_logit = descending_logits[0];
                total_descending_loss += torch::nn::functional::binary_cross_entropy_with_logits(current_descending_logit, descending_probability.detach());

                if (disable_baseline_)
                {
                    total_policy_loss -= (successor_descending_probs.detach() * successor_transition_probs).sum();
                }
                else
                {
                    total_policy_loss -= ((successor_descending_probs.detach() - descending_probability.detach()) * successor_transition_probs).sum();
                }

                for (std::size_t successor_index = 0; successor_index < successor_states.size(); ++successor_index)
                {
                    const auto& successor_state = successor_states[successor_index];
                    const auto successor_is_goal_state = state_space->is_goal_state(successor_state);

                    if (successor_is_goal_state)
                    {
                        const auto& successor_value = successor_values[successor_index];
                        total_goal_value_loss += successor_value.detach() * successor_value;
                        ++num_goal_value_samples;

                        const auto& successor_descending_logit = successor_descending_logits[successor_index];
                        const auto goal_probability = torch::scalar_tensor(1.0, { device });
                        total_goal_descending_loss += torch::nn::functional::binary_cross_entropy_with_logits(successor_descending_logit, goal_probability);
                        ++num_goal_descending_samples;
                    }
                }

                ++num_value_samples;
                ++num_policy_samples;
                ++num_descending_samples;
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

        if (num_descending_samples > 0.0)
        {
            total_descending_loss /= num_descending_samples;
        }

        if (num_goal_value_samples > 0.0)
        {
            total_goal_value_loss /= num_goal_value_samples;
        }

        if (num_goal_descending_samples > 0.0)
        {
            total_goal_descending_loss /= num_goal_descending_samples;
        }

        // The learning rate for each loss should be different, normalize the scale of the gradient so that we can use the same optimizer.

        const auto VALUE_MULTIPLIER = 1.0;
        const auto POLICY_MULTIPLIER = 1.0;
        const auto DESCENDING_MULTIPLIER = 1.0;
        const auto GOAL_VALUE_MULTIPLIER = 1.0;
        const auto GOAL_DESCENDING_MULTIPLIER = 1.0;

        return (VALUE_MULTIPLIER * total_value_loss) + (DESCENDING_MULTIPLIER * total_descending_loss) + (POLICY_MULTIPLIER * total_policy_loss)
               + (GOAL_VALUE_MULTIPLIER * total_goal_value_loss) + (GOAL_DESCENDING_MULTIPLIER * total_goal_descending_loss);
    }

    std::shared_ptr<torch::optim::Optimizer> PolicyIterativeFullProbability::create_optimizer(models::RelationalNeuralNetwork& model)
    {
        return std::make_shared<torch::optim::Adam>(model.parameters(), learning_rate_);
    }

    torch::Tensor PolicyIterativeFullProbability::train_loss(uint32_t epoch, const planners::StateSpaceSampleList& batch, const PolicyBatchOutput& output)
    {
        return this->loss(epoch, batch, output);
    }

    planners::StateSpaceSampleList
    PolicyIterativeFullProbability::get_batch(const std::shared_ptr<datasets::Dataset>& set, uint32_t epoch, uint32_t batch_index, uint32_t batch_size)
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
