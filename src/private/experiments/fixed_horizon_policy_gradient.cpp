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


#include "debug.hpp"
#include "fixed_horizon_policy_gradient.hpp"
#include "returns/cumulative_return_function.hpp"

#include <functional>

namespace experiments
{
    FixedHorizonPolicyGradient::FixedHorizonPolicyGradient(uint32_t batch_size,
                                                           uint32_t max_epochs,
                                                           double learning_rate,
                                                           uint32_t max_horizon,
                                                           double discount,
                                                           const RewardFunction& reward_function) :
        MonteCarloExperiment(batch_size, max_epochs, max_horizon, reward_function),
        learning_rate_(learning_rate),
        discount_(discount) {};

    FixedHorizonPolicyGradient::~FixedHorizonPolicyGradient() {}

    double FixedHorizonPolicyGradient::get_learning_rate() const { return learning_rate_; }

    double FixedHorizonPolicyGradient::get_discount() const { return discount_; }

    torch::Tensor FixedHorizonPolicyGradient::update_transition_gradient(const torch::Tensor gradient,
                                                                         const torch::Tensor update_value,
                                                                         const experiments::TrajectoryStep& step,
                                                                         const double cumulative_discount) const
    {
        // Default implementation is to update the transition probability based solely on the given update_value.
        return gradient + cumulative_discount * update_value * step.transition_probability.log().view(-1)[0];
    }

    torch::Tensor FixedHorizonPolicyGradient::update_value_gradient(const torch::Tensor gradient,
                                                                    const torch::Tensor update_value,
                                                                    const experiments::TrajectoryStep& step,
                                                                    const double cumulative_discount) const
    {
        // Default implementation is to not update the value function.
        return gradient;
    }

    std::pair<torch::Tensor, torch::Tensor>
    FixedHorizonPolicyGradient::loss(const planners::StateSpaceSampleList& batch, const experiments::TrajectoryList& trajectories, bool compute_gradient)
    {
        const auto device = trajectories.at(0).at(0).state_value.device();
        auto total_gradient = torch::scalar_tensor(0.0, device);
        auto total_reward = torch::scalar_tensor(0.0, device);
        auto total_steps = (double) batch.size();

        const CumulativeReturnFunctionImpl cumulative_return;

        for (std::size_t sample_index = 0; sample_index < batch.size(); ++sample_index)
        {
            const auto& trajectory = trajectories.at(sample_index);
            const auto num_steps = trajectory.size();
            auto transition_gradient = torch::scalar_tensor(0.0, device);
            auto value_gradient = torch::scalar_tensor(0.0, device);
            auto trajectory_reward = torch::scalar_tensor(0.0, device);

            double cumulative_discount = 1.0;
            for (std::size_t step_index = 0; step_index < num_steps; ++step_index)
            {
                if (compute_gradient)
                {
                    const auto& step = trajectory[step_index];
                    const auto& sample = batch.at(sample_index);

                    const auto update_value = get_update_value(sample.second->problem, trajectory, step_index);
                    transition_gradient = update_transition_gradient(transition_gradient, update_value, step, cumulative_discount);
                    value_gradient = update_value_gradient(value_gradient, update_value, step, cumulative_discount);
                }

                trajectory_reward += cumulative_return.get(step_index, num_steps - step_index, discount_, trajectory, device);
                cumulative_discount *= discount_;
            }

            total_gradient += (transition_gradient + value_gradient) / (double) num_steps;
            total_reward += trajectory_reward;
            total_steps += num_steps;
        }

        return std::make_pair((total_steps > 0.0) ? (total_gradient / total_steps) : total_gradient,  // Normalize by number of terms.
                              (total_reward / total_steps));                                          // Normalize by number of instances.
    }

    std::shared_ptr<torch::optim::Optimizer> FixedHorizonPolicyGradient::create_optimizer(models::RelationalNeuralNetwork& model)
    {
        return std::make_shared<torch::optim::Adam>(model.parameters(), learning_rate_);
    }

    std::pair<torch::Tensor, torch::Tensor> FixedHorizonPolicyGradient::train_loss(const planners::StateSpaceSampleList& batch,
                                                                                   const experiments::TrajectoryList& trajectories)
    {
        return this->loss(batch, trajectories, true);
    }

    torch::Tensor FixedHorizonPolicyGradient::validation_loss(const planners::StateSpaceSampleList& batch, const experiments::TrajectoryList& trajectories)
    {
        return this->loss(batch, trajectories, false).second;
    }
}  // namespace experiments
