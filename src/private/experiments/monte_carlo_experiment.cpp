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


#include "../algorithms/weisfeiler_leman.hpp"
#include "../datasets/random_dataset.hpp"
#include "../formalism/state.hpp"
#include "../models/utils.hpp"
#include "../planners/generators/lifted_successor_generator.hpp"
#include "monte_carlo_experiment.hpp"

#include <functional>
#include <unordered_set>

namespace experiments
{
    MonteCarloExperiment::MonteCarloExperiment(uint32_t batch_size, uint32_t max_epochs, uint32_t max_horizon, const RewardFunction& reward_function) :
        batch_size_(batch_size),
        max_epochs_(max_epochs),
        max_horizon_(max_horizon),
        reward_function_(reward_function)
    {
    }

    MonteCarloExperiment::~MonteCarloExperiment() {}

    experiments::TrajectoryList MonteCarloExperiment::compute_trajectories(models::RelationalNeuralNetwork& model, const planners::StateSpaceSampleList& batch)
    {
        TrajectoryList batch_trajectories;
        std::vector<formalism::State> batch_current_state(batch.size());

        for (size_t sample_index = 0; sample_index < batch.size(); ++sample_index)
        {
            batch_trajectories.push_back(experiments::Trajectory());
            batch_current_state[sample_index] = batch[sample_index].second->problem->initial;
        }

        // Execute the policy max_horizon steps
        for (size_t step = 0; step < max_horizon_; ++step)
        {
            formalism::StateTransitionsVector transitions_vector;

            // Enumerate all successor states for all batch so that we can use a single forward pass.
            for (size_t sample_index = 0; sample_index < batch.size(); ++sample_index)
            {
                const auto& sample = batch[sample_index];
                const auto& state_space = sample.second;
                const auto& state = batch_current_state[sample_index];

                formalism::StateList successors;

                if (!state_space->is_goal_state(state))
                {
                    for (const auto& transitions : state_space->get_forward_transitions(state))
                    {
                        successors.push_back(transitions->target_state);
                    }
                }

                transitions_vector.push_back(formalism::StateTransitions(state, successors, sample.second->problem));
            }

            // Perform a single forward pass

            const auto output = model.forward(transitions_vector);
            const auto& problem_policy = std::get<0>(output);  // The vector consists of probability-tensors of transitions
            const auto& problem_values = std::get<1>(output);  // The tensor consists of values for all states, but is not grouped
            const auto& reward_function = get_reward_function();

            // Select successor

            for (size_t sample_index = 0; sample_index < batch.size(); ++sample_index)
            {
                const auto& sample = batch[sample_index];
                const auto& state_space = sample.second;
                const auto& state = batch_current_state[sample_index];
                const auto& transitions = transitions_vector.at(sample_index);
                const auto& successors = std::get<1>(transitions);

                if (successors.size() > 0)
                {
                    const auto& policy = problem_policy.at(sample_index).view(-1);
                    const auto& values = problem_values.at(sample_index);
                    const auto policy_select = policy.cumsum(0) - torch::rand(1, policy.device());
                    const auto select_index = std::get<0>(torch::min(policy_select.clamp(0.0).nonzero(), 0, false)).item<int64_t>();

                    const auto selected_transition = policy.select(0, select_index);
                    const auto current_value = values.select(0, 0);
                    const auto successor_value = values.select(0, select_index + 1);
                    const auto successor_state = successors.at(select_index);
                    const auto transition_reward = reward_function->reward(state_space, state, successor_state);

                    batch_trajectories[sample_index].push_back(experiments::TrajectoryStep(state_space,
                                                                                           state,
                                                                                           successor_state,
                                                                                           selected_transition,
                                                                                           current_value,
                                                                                           successor_value,
                                                                                           transition_reward));

                    batch_current_state[sample_index] = successor_state;
                }
            }
        }

        return batch_trajectories;
    }

    RewardFunction MonteCarloExperiment::get_reward_function() const { return reward_function_; }

    void MonteCarloExperiment::fit(models::RelationalNeuralNetwork& model,
                                   const planners::StateSpaceList& training_state_spaces,
                                   const planners::StateSpaceList& validation_state_spaces)
    {
        auto optimizer = create_optimizer(model);
        auto best_validation_solved = -std::numeric_limits<double>::infinity();
        auto best_validation_loss = -std::numeric_limits<double>::infinity();

        const datasets::RandomDataset training_set(training_state_spaces, true, true, true);
        const datasets::RandomDataset validation_set(validation_state_spaces, true, true, true);

        const uint32_t num_train_batches = (training_set.size() / batch_size_) + ((training_set.size() % batch_size_) > 0 ? 1 : 0);
        const uint32_t num_val_batches = (validation_set.size() / batch_size_) + ((validation_set.size() % batch_size_) > 0 ? 1 : 0);

        for (uint32_t epoch = 0; epoch < max_epochs_; epoch++)
        {
            double train_solved = 0.0;
            double train_total = 0.0;
            model.train();

            for (uint32_t batch_index = 0; batch_index < num_train_batches; ++batch_index)
            {
                model.zero_grad();
                const auto batch = training_set.get_range(batch_size_ * batch_index, batch_size_);
                const auto trajectories = compute_trajectories(model, batch);
                const auto output = train_loss(batch, trajectories);
                const auto gradient = -output.first;  // The optimizer performs gradient descent, flip the sign for gradient ascent
                const auto reward = output.second;
                gradient.backward();
                optimizer->step();

                // Print some training information

                int32_t num_solved = 0;
                int32_t num_instances = batch.size();

                for (const auto& trajectory : trajectories)
                {
                    const auto& last_step = trajectory.at(trajectory.size() - 1);

                    if (last_step.state_space->is_goal_state(last_step.successor_state))
                    {
                        ++num_solved;
                    }
                }

                train_solved += num_solved;
                train_total += num_instances;
                const auto train_percent_solved = num_solved / (double) num_instances;
                std::cout << "[" << std::fixed << std::setprecision(2) << train_percent_solved << "]";
                std::cout << "[" << epoch << ", " << batch_index + 1 << "/" << num_train_batches << "] Train: " << reward.item<double>() << std::endl;
            }

            {  // Disable computation of gradients
                torch::NoGradGuard no_grad;
                model.eval();
                auto total_validation_loss = 0.0;  // it would be better to do this on the device, but that crashes for some reason
                auto num_solved = 0;
                for (uint32_t batch_index = 0; batch_index < num_val_batches; ++batch_index)
                {
                    const auto batch = validation_set.get_range(batch_size_ * batch_index, batch_size_);
                    const auto trajectories = compute_trajectories(model, batch);
                    const auto loss = validation_loss(batch, trajectories);
                    total_validation_loss += batch.size() * loss.item<double>();

                    for (const auto& trajectory : trajectories)
                    {
                        const auto& last_step = trajectory.at(trajectory.size() - 1);

                        if (last_step.state_space->is_goal_state(last_step.successor_state))
                        {
                            ++num_solved;
                        }
                    }
                }

                const auto validation_solved = num_solved / (double) validation_set.size();
                std::cout << "[" << std::fixed << std::setprecision(2) << validation_solved << "]";
                const auto validation_loss = total_validation_loss / (double) validation_set.size();
                std::cout << "[" << epoch << "] Validation: " << validation_loss << std::endl;
                models::save_model("latest", model);

                if ((validation_solved > best_validation_solved)
                    || ((validation_solved == best_validation_solved) && (validation_loss >= best_validation_loss)))
                {
                    best_validation_solved = validation_solved;
                    best_validation_loss = validation_loss;
                    models::save_model("best", model);
                    std::cout << "[Info][" << epoch << "] Saved best model (" << std::fixed << std::setprecision(2) << validation_loss << ")" << std::endl;
                }
            }
        }
    }
}  // namespace experiments
