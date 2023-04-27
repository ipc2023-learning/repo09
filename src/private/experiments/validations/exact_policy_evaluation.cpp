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


#include "../policy_evaluation_utils.hpp"
#include "exact_policy_evaluation.hpp"

namespace experiments
{
    ExactPolicyEvaluation::ExactPolicyEvaluation(double discount_factor, uint32_t batch_size) :
        discount_factor_(discount_factor),
        mean_optimal_value_(-1),
        batch_size_(batch_size)
    {
    }

    double ExactPolicyEvaluation::initialize(const planners::StateSpaceList& state_spaces, models::RelationalNeuralNetwork& model)
    {
        double sum_optimal_value = 0.0;
        state_spaces_.insert(state_spaces_.end(), state_spaces.begin(), state_spaces.end());

        for (const auto& state_space : state_spaces)
        {
            const auto transition_probabilities = create_transition_probabilities(state_space);
            const auto batches_indices = create_batches(state_space->problem, transition_probabilities, batch_size_);
            const auto& batches = batches_indices.first;
            const auto& indices = batches_indices.second;

            validation_transition_probabilities_.push_back(std::move(transition_probabilities));
            validation_batches_.push_back(std::move(batches));
            validation_indices_.push_back(std::move(indices));

            double problem_sum_optimal_value = 0.0;

            for (const auto& state : state_space->get_states())
            {
                if (state_space->is_dead_end_state(state))
                {
                    problem_sum_optimal_value += 1.0 / (1.0 - discount_factor_);
                }
                else
                {
                    problem_sum_optimal_value += (std::pow(discount_factor_, state_space->get_distance_to_goal_state(state)) - 1.0) / (discount_factor_ - 1.0);
                }
            }

            sum_optimal_value += problem_sum_optimal_value / (double) state_space->num_states();
        }

        mean_optimal_value_ = sum_optimal_value / (double) state_spaces.size();
        return mean_optimal_value_;
    }

    double ExactPolicyEvaluation::evaluate(models::RelationalNeuralNetwork& model)
    {
        auto sum_validation_loss = 0.0;
        for (std::size_t index = 0; index < state_spaces_.size(); ++index)
        {
            auto& transition_probabilities = validation_transition_probabilities_[index];
            const auto& batches = validation_batches_[index];
            const auto& indices = validation_indices_[index];

            update_transition_probabilities(batches, indices, transition_probabilities, model);
            const auto values_vector = compute_policy_evaluation(transition_probabilities, discount_factor_);
            sum_validation_loss += std::accumulate(values_vector.begin(), values_vector.end(), 0.0) / values_vector.size();
        }

        return sum_validation_loss / (double) state_spaces_.size();
    }
}  // namespace experiments
