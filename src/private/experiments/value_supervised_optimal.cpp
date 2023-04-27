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


#include "value_supervised_optimal.hpp"

namespace experiments
{
    torch::Tensor SupervisedOptimal::loss(const planners::StateSpaceSampleList& batch, const std::pair<torch::Tensor, torch::Tensor>& output)
    {
        const auto output_values = output.first.view(-1);
        const auto output_dead_ends = output.second.view(-1);

        std::vector<double> vector_target_values;
        std::vector<double> vector_target_dead_ends;

        for (const auto& sample : batch)
        {
            const auto& state = sample.first;
            const auto& state_space = sample.second;
            vector_target_values.push_back((double) state_space->get_distance_to_goal_state(state));
            vector_target_dead_ends.push_back(state_space->is_dead_end_state(state) ? 1.0 : 0.0);
        }

        const auto target_values = torch::tensor(vector_target_values).to(output_values.device());
        const auto target_dead_ends = torch::tensor(vector_target_dead_ends).to(output_values.device());
        const auto target_values_mask = -1.0 * (target_dead_ends - 1.0);

        const auto loss_values = (output_values - target_values).abs() * target_values_mask;
        const auto aggregated_loss_values = loss_values.sum() / target_values_mask.sum();
        const auto aggregated_loss_dead_ends = torch::nn::functional::binary_cross_entropy_with_logits(output_dead_ends, target_dead_ends);

        return aggregated_loss_values + aggregated_loss_dead_ends;
    }

    torch::Tensor SupervisedOptimal::train_loss(const planners::StateSpaceSampleList& batch, const std::pair<torch::Tensor, torch::Tensor>& output)
    {
        return this->loss(batch, output);
    }

    torch::Tensor SupervisedOptimal::validation_loss(const planners::StateSpaceSampleList& batch, const std::pair<torch::Tensor, torch::Tensor>& output)
    {
        return this->loss(batch, output);
    }
}  // namespace experiments
