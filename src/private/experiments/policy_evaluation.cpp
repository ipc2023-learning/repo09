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
#include "policy_evaluation.hpp"
#include "policy_evaluation_utils.hpp"

#include <chrono>
#include <limits>
#include <unordered_map>

namespace timer = std::chrono;

namespace experiments
{
    void
    PolicyEvaluation::fit(models::RelationalNeuralNetwork& model, const planners::StateSpaceList& training_set, const planners::StateSpaceList& validation_set)
    {
        auto best_validation_loss = std::numeric_limits<double>::infinity();
        torch::optim::Adam optimizer(model.parameters(), learning_rate_);
        const auto fit_start = timer::high_resolution_clock::now();

        std::vector<TransitionProbabilities> train_transition_probabilities;
        std::vector<std::vector<formalism::StateTransitionsVector>> train_batches;
        std::vector<std::vector<std::vector<std::size_t>>> train_indices;
        std::vector<std::vector<double>> train_weights;

        for (const auto& state_space : training_set)
        {
            const auto transition_probabilities = create_transition_probabilities(state_space);
            const auto batches_indices = create_batches(state_space->problem, transition_probabilities, batch_size_);
            const auto weights = state_space->get_distance_to_goal_state_weights();
            const auto& batches = batches_indices.first;
            const auto& indices = batches_indices.second;

            train_transition_probabilities.push_back(std::move(transition_probabilities));
            train_batches.push_back(std::move(batches));
            train_indices.push_back(std::move(indices));
            train_weights.push_back(std::move(weights));
        }

        std::vector<TransitionProbabilities> validation_transition_probabilities;
        std::vector<std::vector<formalism::StateTransitionsVector>> validation_batches;
        std::vector<std::vector<std::vector<std::size_t>>> validation_indices;

        double total_optimal_value = 0.0;

        for (const auto& state_space : validation_set)
        {
            const auto transition_probabilities = create_transition_probabilities(state_space);
            const auto batches_indices = create_batches(state_space->problem, transition_probabilities, batch_size_);
            const auto& batches = batches_indices.first;
            const auto& indices = batches_indices.second;

            validation_transition_probabilities.push_back(std::move(transition_probabilities));
            validation_batches.push_back(std::move(batches));
            validation_indices.push_back(std::move(indices));

            double problem_total_optimal_value = 0.0;

            for (const auto& state : state_space->get_states())
            {
                if (state_space->is_dead_end_state(state))
                {
                    const auto optimal_value = 1.0 / (1.0 - discount_factor_);
                    problem_total_optimal_value += optimal_value;
                }
                else
                {
                    const auto optimal_value = (std::pow(discount_factor_, state_space->get_distance_to_goal_state(state)) - 1.0) / (discount_factor_ - 1.0);
                    problem_total_optimal_value += optimal_value;
                }
            }

            total_optimal_value += problem_total_optimal_value / (double) state_space->num_states();
        }

        const auto average_optimal_value = total_optimal_value / (double) validation_set.size();

        std::cout << std::fixed << std::setprecision(5);
        std::cout << "Optimal validation loss: " << average_optimal_value << std::endl;

        for (uint32_t epoch = 0; epoch < max_epochs_; ++epoch)
        {
            const auto epoch_start = timer::high_resolution_clock::now();

            // Training

            const auto train_start = timer::high_resolution_clock::now();

            model.zero_grad();

            for (std::size_t index = 0; index < training_set.size(); ++index)
            {
                const auto instance_start = timer::high_resolution_clock::now();

                auto& transition_probabilities = train_transition_probabilities[index];
                const auto& batches = train_batches[index];
                const auto& weights = train_weights[index];
                const auto& state_space = training_set[index];
                const auto& state_index_list = train_indices[index];

                const auto update_start = timer::high_resolution_clock::now();
                model.eval();
                update_transition_probabilities(batches, state_index_list, transition_probabilities, model);
                const auto update_end = timer::high_resolution_clock::now();

                const auto compute_start = timer::high_resolution_clock::now();
                const auto optimal_vector = compute_optimal_policy_evaluation(state_space, discount_factor_);
                const auto value_vector = compute_policy_evaluation(transition_probabilities, discount_factor_);
                const auto compute_end = timer::high_resolution_clock::now();

                const auto& value_tensor = torch::tensor(value_vector).to(model.device());

                model.train();

                double instance_loss = 0.0;

                for (std::size_t batch_index = 0; batch_index < batches.size(); ++batch_index)
                {
                    auto loss = torch::scalar_tensor(0.0).to(model.device());
                    bool any_loss = false;
                    const auto& batch = batches[batch_index];
                    const auto& state_indices = state_index_list[batch_index];

                    const auto batch_output = model.forward(batch);
                    const auto policy_output = std::get<0>(batch_output);
                    const auto value_output = std::get<1>(batch_output);

                    for (std::size_t output_index = 0; output_index < policy_output.size(); ++output_index)
                    {
                        any_loss = true;
                        const auto state_index = state_indices[output_index];
                        const auto state_baseline = value_vector[state_index] - 1.0;
                        const auto state = std::get<0>(batch[output_index]);
                        const auto successor_states = std::get<1>(batch[output_index]);

                        std::vector<double> successor_weights;
                        for (const auto& successor_state : successor_states)
                        {
                            successor_weights.push_back(disable_balancing_ ? 1.0 : weights[state_space->get_distance_to_goal_state(successor_state)]);
                        }

                        const auto neighbor_indices_vector = transition_probabilities.neighbor_indices[state_index];
                        const auto neighbor_indices_tensor = torch::tensor(neighbor_indices_vector).to(model.device(), torch::kInt64, true, false);
                        const auto neighbor_weights_tensor = torch::tensor(successor_weights).to(model.device(), true, false);
                        const auto neighbor_values = discount_factor_ * value_tensor.index_select(0, neighbor_indices_tensor).detach();
                        const auto state_space_size = (double) transition_probabilities.states.size();
                        const auto normalization = (1.0 / state_space_size);

                        if (disable_baseline_)
                        {
                            loss += normalization * (neighbor_weights_tensor * neighbor_values * policy_output[output_index].view(-1)).sum();
                        }
                        else
                        {
                            // TODO: Check if baseline is correctly implemented according to pseudo-code.
                            loss += normalization * (neighbor_weights_tensor * (neighbor_values - state_baseline) * policy_output[output_index].view(-1)).sum();
                        }

                        if (!disable_value_regularization_)
                        {
                            const auto value_optimal = optimal_vector[state_index];
                            const auto value_target = value_optimal;

                            loss += normalization * (value_target - value_output[output_index].view(-1)[0]).abs();

                            for (std::size_t offset = 0; offset < neighbor_indices_vector.size(); ++offset)
                            {
                                const auto neighbor_index = neighbor_indices_vector[offset];

                                if (value_vector[neighbor_index] == 0.0)
                                {
                                    loss += normalization * value_output[output_index].view(-1)[1 + offset].abs();
                                }
                            }
                        }
                    }

                    if (any_loss)
                    {
                        instance_loss += loss.item<double>();
                        loss.backward();
                    }
                }

                const auto update_ds = timer::duration_cast<timer::milliseconds>(update_end - update_start).count() / 1000.0;
                const auto compute_ds = timer::duration_cast<timer::milliseconds>(compute_end - compute_start).count() / 1000.0;

                const auto instance_end = timer::high_resolution_clock::now();
                const auto instance_ds = timer::duration_cast<timer::milliseconds>(instance_end - instance_start).count() / 1000.0;
                const auto instance_ts = timer::duration_cast<timer::milliseconds>(instance_end - fit_start).count() / 1000.0;

                std::cout << std::fixed << std::setprecision(5);
                std::cout << "[" << epoch << ", " << index + 1 << "/" << training_set.size() << "] Train: " << instance_loss;

                std::cout << std::fixed << std::setprecision(2);
                std::cout << " (d = " << instance_ds << " s, t = " << instance_ts << " s, whereof ud = " << update_ds << " s, cd = " << compute_ds << " s)"
                          << std::endl;
            }

            optimizer.step();

            const auto train_end = timer::high_resolution_clock::now();
            const auto train_ds = timer::duration_cast<timer::milliseconds>(train_end - train_start).count() / 1000.0;
            const auto train_ts = timer::duration_cast<timer::milliseconds>(train_end - fit_start).count() / 1000.0;
            std::cout << std::fixed << std::setprecision(2) << "[" << epoch << "] Train: d = " << train_ds << " s, t = " << train_ts << " s" << std::endl;

            // Validation

            const auto val_start = timer::high_resolution_clock::now();
            model.eval();

            auto total_validation_loss = 0.0;
            for (std::size_t index = 0; index < validation_set.size(); ++index)
            {
                auto& transition_probabilities = validation_transition_probabilities[index];
                const auto& batches = validation_batches[index];
                const auto& indices = validation_indices[index];

                update_transition_probabilities(batches, indices, transition_probabilities, model);
                const auto values_vector = compute_policy_evaluation(transition_probabilities, discount_factor_);
                total_validation_loss += std::accumulate(values_vector.begin(), values_vector.end(), 0.0) / values_vector.size();
            }
            const auto validation_loss = total_validation_loss / (double) validation_set.size();
            std::cout << std::fixed << std::setprecision(5) << "[" << epoch << "] Validation: " << validation_loss << " ("
                      << (validation_loss - average_optimal_value) << ")" << std::endl;

            const auto val_end = timer::high_resolution_clock::now();
            const auto val_ds = timer::duration_cast<timer::milliseconds>(val_end - val_start).count() / 1000.0;
            const auto val_ts = timer::duration_cast<timer::milliseconds>(val_end - fit_start).count() / 1000.0;
            std::cout << std::fixed << std::setprecision(2) << "[" << epoch << "] Validation: d = " << val_ds << " s, t = " << val_ts << std::endl;

            const auto epoch_end = timer::high_resolution_clock::now();
            const auto epoch_ds = timer::duration_cast<timer::milliseconds>(epoch_end - epoch_start).count() / 1000.0;
            const auto epoch_ts = timer::duration_cast<timer::milliseconds>(epoch_end - fit_start).count() / 1000.0;
            std::cout << std::fixed << std::setprecision(2) << "[" << epoch << "] Epoch: d = " << epoch_ds << " s, t = " << epoch_ts << std::endl;

            if (validation_loss < best_validation_loss)
            {
                best_validation_loss = validation_loss;
                std::cout << "[" << epoch << "] Saved new best model" << std::endl;
                models::save_model("best", model);
            }

            models::save_model("latest", model);
        }
    }
}  // namespace experiments
