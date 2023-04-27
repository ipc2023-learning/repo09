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


#include "value_selfsupervised_suboptimal.hpp"

namespace experiments
{
    torch::Tensor SelfsupervisedSuboptimal::loss(const planners::StateSpaceSampleList& batch, const std::pair<torch::Tensor, torch::Tensor>& output)
    {
        const auto output_values = output.first.view(-1);
        const auto output_dead_ends = output.second.view(-1);
        const auto device = output_values.device();

        auto total_goal_loss = torch::scalar_tensor(0.0).to(device);
        auto total_non_goal_loss = torch::scalar_tensor(0.0).to(device);
        auto total_dead_end_loss = torch::scalar_tensor(0.0).to(device);
        auto num_goal_states = 0.0;
        auto num_non_goal_states = 0.0;
        auto num_dead_end_states = 0.0;

        // TODO: this algorithm is essentially the same as Value Iteration (VI), and in VI it is common to prioritize states that has a high error to its best
        // successor; try it out and see if it helps.

        // TODO: try to see if we can use curriculum learning: use states k=1 steps from the goal until some error metric has been met, then increment k by 1.

        for (uint32_t state_index = 0; state_index < batch.size(); /* increment at the end */)
        {
            const auto& state = batch[state_index].first;
            const auto& state_space = batch[state_index].second;

            uint32_t num_successors = 0;

            if (!state_space->is_dead_end_state(state) && !state_space->is_goal_state(state))
            {
                for (const auto& transition : state_space->get_forward_transitions(state))
                {
                    const auto& successor = transition->target_state;

                    if (!state_space->is_dead_end_state(successor))
                    {
                        ++num_successors;
                    }
                }
            }

            const auto target_dead_end = torch::scalar_tensor(state_space->is_dead_end_state(state) ? 1.0 : 0.0).to(device);
            const auto dead_end_loss = torch::nn::functional::binary_cross_entropy_with_logits(output_dead_ends[state_index], target_dead_end);
            total_dead_end_loss += dead_end_loss;
            ++num_dead_end_states;  // all states in the batch are used to train the dead-end classifier

            if (!state_space->is_dead_end_state(state))
            {
                if (state_space->is_goal_state(state))
                {
                    ++num_goal_states;
                    const auto goal_loss = output_values[state_index].abs();
                    total_goal_loss += goal_loss;
                }
                else
                {
                    ++num_non_goal_states;
                    const auto target_value = (double) state_space->get_distance_to_goal_state(state);
                    const auto output_value = output_values[state_index];

                    const auto successors_start_index_incl = state_index + 1;
                    const auto successors_end_index_excl = successors_start_index_incl + num_successors;
                    const auto successor_values = output_values.slice(0, successors_start_index_incl, successors_end_index_excl).abs();
                    const auto min_successor_value = successor_values.min();

                    const auto successor_loss = torch::clamp(1.0 + (min_successor_value - output_value), 0.0);
                    const auto lower_bound_loss = torch::clamp(target_value - output_value, 0.0);
                    const auto upper_bound_loss = torch::clamp(output_value - approximation_factor_ * target_value, 0.0);

                    total_non_goal_loss += successor_loss + lower_bound_loss + upper_bound_loss;
                }
            }

            state_index += num_successors + 1;
        }

        if (num_goal_states > 0.0)
        {
            total_goal_loss /= num_goal_states;
        }

        if (num_non_goal_states > 0.0)
        {
            total_non_goal_loss /= num_non_goal_states;
        }

        if (num_dead_end_states > 0.0)
        {
            total_dead_end_loss /= num_dead_end_states;
        }

        return total_goal_loss + total_non_goal_loss + total_dead_end_loss;
    }

    torch::Tensor SelfsupervisedSuboptimal::train_loss(const planners::StateSpaceSampleList& batch, const std::pair<torch::Tensor, torch::Tensor>& output)
    {
        return this->loss(batch, output);
    }

    torch::Tensor SelfsupervisedSuboptimal::validation_loss(const planners::StateSpaceSampleList& batch, const std::pair<torch::Tensor, torch::Tensor>& output)
    {
        return this->loss(batch, output);
    }

    planners::StateSpaceSampleList SelfsupervisedSuboptimal::get_batch(const datasets::Dataset& set, uint32_t batch_index, uint32_t batch_size)
    {
        planners::StateSpaceSampleList batch;

        uint32_t max_index_excl = std::min((uint32_t) set.size(), batch_index * batch_size + batch_size);
        for (uint32_t index = (batch_index * batch_size); index < max_index_excl; ++index)
        {
            const auto sample = set.get(index);
            const auto& state = sample.first;
            const auto& state_space = sample.second;
            batch.push_back(sample);

            // only consider successors for solvable states and non-goal states
            if (!state_space->is_dead_end_state(state) && !state_space->is_goal_state(state))
            {
                for (const auto& transition : state_space->get_forward_transitions(state))
                {
                    const auto& successor = transition->target_state;

                    if (!state_space->is_dead_end_state(successor))
                    {
                        // only add solvable successor states since we want to successors to be solvable in the learned policy
                        batch.push_back(std::make_pair(successor, state_space));
                    }
                }
            }
        }

        return batch;
    }
}  // namespace experiments
