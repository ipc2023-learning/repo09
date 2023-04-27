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
#include "policy_supervised_optimal.hpp"
#include "torch/torch.h"

#include <chrono>
#include <limits>
#include <unordered_map>

namespace timer = std::chrono;

namespace experiments
{
    std::shared_ptr<torch::optim::Optimizer> PolicySupervisedOptimal::create_optimizer(models::RelationalNeuralNetwork& model)
    {
        return std::make_shared<torch::optim::Adam>(model.parameters(), learning_rate_);
    }

    torch::Tensor PolicySupervisedOptimal::train_loss(uint32_t epoch, const planners::StateSpaceSampleList& batch, const PolicyBatchOutput& output)
    {
        const auto device = output.policies.at(0).device();

        auto policy_loss = torch::scalar_tensor(0.0).to(device);
        auto value_loss = torch::scalar_tensor(0.0).to(device);
        auto goal_loss = torch::scalar_tensor(0.0).to(device);

        auto policy_size = 0.0;
        auto value_size = 0.0;
        auto goal_size = 0.0;

        for (std::size_t index = 0; index < batch.size(); ++index)
        {
            const auto& sample = batch[index];
            const auto& state = sample.first;
            const auto& state_space = sample.second;
            const auto value = state_space->get_distance_to_goal_state(state);

            // Policy loss

            if (value > 0)
            {
                const auto& successor_states = std::get<1>(output.state_successors[index]);
                std::vector<double> labels;

                for (const auto& successor_state : successor_states)
                {
                    const auto successor_value = state_space->get_distance_to_goal_state(successor_state);
                    labels.push_back(successor_value < value ? 1.0 : 0.0);
                }

                const auto mask = torch::tensor(labels).to(device);
                const auto policy_prediction = output.policies.at(index).view(-1);
                policy_loss += 1.0 - (mask * policy_prediction).sum();
                ++policy_size;
            }

            // Value loss

            if (!disable_value_regularization_ && (value > 0))
            {
                const auto value_predictions = output.values.at(index).view(-1);
                const auto value_prediction = value_predictions[0];
                value_loss += (value - value_prediction).abs();
                ++value_size;
            }

            // Goal loss

            if (!disable_value_regularization_ && (value == 0))
            {
                const auto value_predictions = output.values.at(index).view(-1);
                const auto value_prediction = value_predictions[0];
                goal_loss += value_prediction.abs();
                ++goal_size;
            }

            // TODO: Handle dead-ends
        }

        if (policy_size > 0.0)
        {
            policy_loss /= policy_size;
        }

        if (value_size > 0.0)
        {
            value_loss /= value_size;
        }

        if (goal_size > 0.0)
        {
            goal_loss /= goal_size;
        }

        return policy_loss + value_loss + goal_loss;
    }
}  // namespace experiments
