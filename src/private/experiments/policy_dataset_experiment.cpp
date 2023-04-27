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


#include "../datasets/balanced_dataset.hpp"
#include "../datasets/random_dataset.hpp"
#include "../models/utils.hpp"
#include "policy_dataset_experiment.hpp"
#include "policy_evaluation_utils.hpp"
#include "validations/exact_policy_evaluation.hpp"

#include <chrono>

namespace timer = std::chrono;

namespace experiments
{
    PolicyBatchOutput forward_policy_in_chunks(models::RelationalNeuralNetwork& model,
                                               PolicySamplingMethod sampling_method,
                                               const planners::StateSpaceSampleList& batch,
                                               uint32_t max_chunk_size,
                                               int32_t step)
    {
        std::vector<torch::Tensor> policies;
        std::vector<torch::Tensor> values;
        std::vector<torch::Tensor> dead_ends;
        std::vector<std::pair<formalism::State, formalism::StateList>> state_successors;

        formalism::StateTransitionsVector current_chunk;
        std::size_t current_chunk_size = 0;

        for (const auto& sample : batch)
        {
            const auto& state = sample.first;
            const auto& state_space = sample.second;
            const auto& forward_transitions = state_space->get_forward_transitions(state);
            const auto sample_size = forward_transitions.size() + 1;

            if ((current_chunk_size > 0) && ((current_chunk_size + sample_size) > max_chunk_size))
            {
                const auto& chunk_output = model.forward(current_chunk);
                const auto& chunk_policies = std::get<0>(chunk_output);
                const auto& chunk_values = std::get<1>(chunk_output);
                const auto& chunk_dead_ends = std::get<2>(chunk_output);

                policies.insert(policies.end(), chunk_policies.begin(), chunk_policies.end());
                values.insert(values.end(), chunk_values.begin(), chunk_values.end());
                dead_ends.insert(dead_ends.end(), chunk_dead_ends.begin(), chunk_dead_ends.end());

                current_chunk.clear();
                current_chunk_size = 0;
            }

            std::vector<formalism::State> successors;
            for (const auto& forward_transition : forward_transitions)
            {
                successors.push_back(forward_transition->target_state);
            }
            state_successors.push_back(std::make_pair(state, successors));

            current_chunk.push_back(formalism::StateTransitions(state, std::move(successors), state_space->problem));
            current_chunk_size += sample_size;
        }

        if (current_chunk_size > 0)
        {
            const auto& chunk_output = model.forward(current_chunk);
            const auto& chunk_policies = std::get<0>(chunk_output);
            const auto& chunk_values = std::get<1>(chunk_output);
            const auto& chunk_dead_ends = std::get<2>(chunk_output);

            policies.insert(policies.end(), chunk_policies.begin(), chunk_policies.end());
            values.insert(values.end(), chunk_values.begin(), chunk_values.end());
            dead_ends.insert(dead_ends.end(), chunk_dead_ends.begin(), chunk_dead_ends.end());

            current_chunk.clear();
            current_chunk_size = 0;
        }

        // Sample successors based on policy

        std::vector<int64_t> sampled_successor_indices;
        std::vector<torch::Tensor> sampled_successor_probabilities;
        std::vector<torch::Tensor> sampled_successor_values;
        std::vector<torch::Tensor> sampled_successor_dead_ends;
        std::vector<formalism::State> sampled_successor_states;

        for (std::size_t index = 0; index < policies.size(); ++index)
        {
            const auto policy_output = policies[index].view(-1);
            const auto value_output = values[index].view(-1);
            const auto dead_end_output = dead_ends[index].view(-1);
            const auto& successor_states = state_successors[index];

            if (successor_states.second.size() > 0)
            {
                int64_t transition_index;

                if (sampling_method == PolicySamplingMethod::Policy)
                {
                    const auto distribution_shifts = policy_output.cumsum(0) - torch::rand(1, policy_output.device());
                    transition_index = std::get<0>(torch::min(distribution_shifts.clamp(0.0).nonzero(), 0, false)).item<int64_t>();
                }
                else if (sampling_method == PolicySamplingMethod::Uniform)
                {
                    transition_index = torch::randint(policy_output.numel(), { 1 }).item<int64_t>();
                }
                else
                {
                    throw std::runtime_error("sampling method not implemented");
                }

                const auto sampled_successor_probability = policy_output[transition_index];
                const auto sampled_successor_value = value_output[transition_index + 1];
                const auto sampled_successor_dead_end = dead_end_output[transition_index + 1];
                const auto& sampled_successor_state = successor_states.second[transition_index];

                sampled_successor_indices.push_back(transition_index);
                sampled_successor_probabilities.push_back(sampled_successor_probability);
                sampled_successor_values.push_back(sampled_successor_value);
                sampled_successor_dead_ends.push_back(sampled_successor_dead_end);
                sampled_successor_states.push_back(sampled_successor_state);
            }
            else
            {
                sampled_successor_indices.push_back(-1);
                sampled_successor_probabilities.push_back(torch::empty(0));
                sampled_successor_values.push_back(torch::empty(0));
                sampled_successor_dead_ends.push_back(torch::empty(0));
                sampled_successor_states.push_back(nullptr);
            }
        }

        return PolicyBatchOutput(state_successors,
                                 policies,
                                 values,
                                 dead_ends,
                                 sampled_successor_indices,
                                 sampled_successor_probabilities,
                                 sampled_successor_values,
                                 sampled_successor_dead_ends,
                                 sampled_successor_states,
                                 step);
    }

    torch::Tensor PolicyDatasetExperiment::train_loss(uint32_t epoch, const planners::StateSpaceSampleList& batch, const PolicyBatchOutput& output)
    {
        throw std::runtime_error("not implemented");
    }

    torch::Tensor PolicyDatasetExperiment::train_loss(uint32_t epoch,
                                                      const planners::StateSpaceSampleList& batch,
                                                      const PolicyBatchOutput& output,
                                                      models::RelationalNeuralNetwork& model)
    {
        return train_loss(epoch, batch, output);
    }

    planners::StateSpaceSampleList
    PolicyDatasetExperiment::get_batch(const std::shared_ptr<datasets::Dataset>& set, uint32_t epoch, uint32_t batch_index, uint32_t batch_size)
    {
        return set->get_range(batch_index * batch_size, batch_size);
    }

    uint32_t PolicyDatasetExperiment::get_batch_size() const { return batch_size_; }

    double PolicyDatasetExperiment::get_discount_factor() const { return discount_factor_; }

    void PolicyDatasetExperiment::fit(models::RelationalNeuralNetwork& model,
                                      const planners::StateSpaceList& training_state_spaces,
                                      const planners::StateSpaceList& validation_state_spaces)
    {
        auto optimizer = create_optimizer(model);
        auto best_validation_loss = std::numeric_limits<double>::infinity();
        const auto time_fit_start = timer::high_resolution_clock::now();
        auto time_divergence_start = timer::high_resolution_clock::now();
        auto time_last_validation = time_fit_start;

        ExactPolicyEvaluation validation(get_discount_factor(), batch_size_);
        const double validation_optimal_value = validation.initialize(validation_state_spaces, model);

        uint64_t num_divergence_samples = 0;
        uint64_t num_training_states = 0;
        std::map<planners::StateSpace, std::vector<uint32_t>> training_distributions;
        for (const auto& state_space : training_state_spaces)
        {
            training_distributions.insert(std::make_pair(state_space, std::vector<uint32_t>(state_space->num_states())));
            num_training_states += (uint64_t) ((state_space->num_states() - state_space->num_goal_states()) - state_space->num_dead_end_states());
        }

        std::cout << std::fixed << std::setprecision(5);
        std::cout << "Optimal validation loss: " << validation_optimal_value << std::endl;

        std::shared_ptr<datasets::Dataset> training_set = nullptr;

        if (disable_balancing_)
        {
            training_set = std::make_shared<datasets::RandomDataset>(training_state_spaces, true, true, true);
        }
        else
        {
            training_set = std::make_shared<datasets::BalancedDataset>(training_state_spaces, true, true, true);
        }

        for (uint32_t epoch = 0; epoch < max_epochs_; epoch++)
        {
            const auto time_epoch_start = timer::high_resolution_clock::now();

            // Training

            const uint32_t num_train_batches = (training_set->size() / batch_size_) + ((training_set->size() % batch_size_) > 0 ? 1 : 0);

            for (uint32_t batch_index = 0; batch_index < num_train_batches; ++batch_index)
            {
                const auto time_batch_start = timer::high_resolution_clock::now();
                model.zero_grad();
                auto batch = get_batch(training_set, epoch, batch_index, batch_size_);

                for (uint32_t step = 0; step < trajectory_length_; ++step)
                {
                    const auto time_step_start = timer::high_resolution_clock::now();

                    if (batch.size() > 0)
                    {
                        model.train();
                        const auto output = forward_policy_in_chunks(model, sampling_method_, batch, chunk_size_, step);
                        const auto loss = train_loss(epoch, batch, output, model);

                        if (!loss.requires_grad())
                        {
                            continue;
                        }

                        loss.backward();
                        optimizer->step();
                        const auto batch_loss = loss.item<double>();

                        log_timings("Train", time_step_start, time_fit_start, epoch, batch_index, num_train_batches, step, trajectory_length_, batch_loss);

                        if (std::isnan(batch_loss) || std::isinf(batch_loss))
                        {
                            std::cout << "[" << epoch << ", " << batch_index + 1 << "/" << num_train_batches << "] Train: Update is either NAN or INF, aborting"
                                      << std::endl;
                            return;
                        }

                        if (std::abs(batch_loss) > 100000000000000000.0)
                        {
                            std::cout << "[" << epoch << ", " << batch_index + 1 << "/" << num_train_batches << "] Train: Update is very large, aborting"
                                      << std::endl;
                            return;
                        }

                        // Update batch based on the sampled successor.

                        planners::StateSpaceSampleList next_batch;

                        for (uint32_t sample_index = 0; sample_index < batch.size(); ++sample_index)
                        {
                            const auto& sample = batch[sample_index];
                            const auto& state = sample.first;
                            const auto& state_space = sample.second;
                            const auto& successor_state = output.sampled_successor_states[sample_index];

                            if ((successor_state != nullptr) && !state_space->is_goal_state(successor_state))
                            {
                                next_batch.push_back(std::make_pair(successor_state, state_space));
                            }

                            auto& distribution_vector = training_distributions.at(state_space);
                            distribution_vector[state_space->get_unique_index_of_state(state)] += 1;
                            num_divergence_samples += 1;
                        }

                        if (num_divergence_samples >= (10 * num_training_states))
                        {
                            log_and_reset_divergence(training_distributions,
                                                     num_training_states,
                                                     num_divergence_samples,
                                                     time_divergence_start,
                                                     time_fit_start,
                                                     epoch);

                            time_divergence_start = timer::high_resolution_clock::now();
                            num_divergence_samples = 0;
                        }

                        batch = std::move(next_batch);
                    }
                    else
                    {
                        std::cout << "[" << epoch << ", " << batch_index + 1 << "/" << num_train_batches << "] Train: Empty batch" << std::endl;
                        break;
                    }

                    const auto time_now = timer::high_resolution_clock::now();
                    const auto time_last_validation_delta = timer::duration_cast<timer::seconds>(time_now - time_last_validation).count();

                    if (time_last_validation_delta >= 120)
                    {
                        model.eval();
                        const auto validation_policy_value = validation.evaluate(model);

                        log_timings("Validation", time_now, time_fit_start, epoch, validation_policy_value);

                        if (validation_policy_value < best_validation_loss)
                        {
                            best_validation_loss = validation_policy_value;
                            std::cout << "[" << epoch << "] Saved new best model" << std::endl;
                            models::save_model("best", model);
                        }

                        models::save_model("latest", model);

                        time_last_validation = timer::high_resolution_clock::now();
                    }
                }

                log_timings("Batch", time_batch_start, time_fit_start, epoch, batch_index, num_train_batches);
            }

            log_timings("Epoch", time_epoch_start, time_fit_start, epoch);
        }
    }

    void PolicyDatasetExperiment::log_timings(const std::string& message,
                                              std::chrono::high_resolution_clock::time_point inner_start,
                                              std::chrono::high_resolution_clock::time_point outer_start,
                                              uint32_t epoch)
    {
        const auto time_end = timer::high_resolution_clock::now();
        const auto time_delta = timer::duration_cast<timer::milliseconds>(time_end - inner_start).count() / 1000.0;
        const auto time_total = timer::duration_cast<timer::milliseconds>(time_end - outer_start).count() / 1000.0;
        std::cout << std::fixed << std::setprecision(2) << "[" << epoch << "] " << message << ": d = " << time_delta << " s, t = " << time_total << std::endl;
    }

    void PolicyDatasetExperiment::log_timings(const std::string& message,
                                              std::chrono::high_resolution_clock::time_point inner_start,
                                              std::chrono::high_resolution_clock::time_point outer_start,
                                              uint32_t epoch,
                                              uint32_t index,
                                              uint32_t length)
    {
        const auto time_end = timer::high_resolution_clock::now();
        const auto time_delta = timer::duration_cast<timer::milliseconds>(time_end - inner_start).count() / 1000.0;
        const auto time_total = timer::duration_cast<timer::milliseconds>(time_end - outer_start).count() / 1000.0;
        std::cout << std::fixed << std::setprecision(2) << "[" << epoch << ", " << (index + 1) << "/" << length << "] " << message << ": d = " << time_delta
                  << " s, t = " << time_total << std::endl;
    }

    void PolicyDatasetExperiment::log_timings(const std::string& message,
                                              std::chrono::high_resolution_clock::time_point inner_start,
                                              std::chrono::high_resolution_clock::time_point outer_start,
                                              uint32_t epoch,
                                              uint32_t outer_index,
                                              uint32_t outer_length,
                                              uint32_t inner_index,
                                              uint32_t inner_length,
                                              double value)
    {
        const auto time_end = timer::high_resolution_clock::now();
        const auto time_delta = timer::duration_cast<timer::milliseconds>(time_end - inner_start).count() / 1000.0;
        const auto time_total = timer::duration_cast<timer::milliseconds>(time_end - outer_start).count() / 1000.0;
        std::cout << std::fixed << std::setprecision(2) << "[" << epoch << ", " << (outer_index + 1) << "/" << outer_length << ", " << (inner_index + 1) << "/"
                  << inner_length << "] " << message << ": " << std::setprecision(5) << value << std::setprecision(2) << ", d = " << time_delta
                  << " s, t = " << time_total << std::endl;
    }

    void PolicyDatasetExperiment::log_timings(const std::string& message,
                                              std::chrono::high_resolution_clock::time_point inner_start,
                                              std::chrono::high_resolution_clock::time_point outer_start,
                                              uint32_t epoch,
                                              double value)
    {
        const auto time_end = timer::high_resolution_clock::now();
        const auto time_delta = timer::duration_cast<timer::milliseconds>(time_end - inner_start).count() / 1000.0;
        const auto time_total = timer::duration_cast<timer::milliseconds>(time_end - outer_start).count() / 1000.0;
        std::cout << std::fixed << std::setprecision(2) << "[" << epoch << "] " << message << ": " << std::setprecision(5) << value << std::setprecision(2)
                  << ", d = " << time_delta << " s, t = " << time_total << " s" << std::endl;
    }

    void PolicyDatasetExperiment::log_and_reset_divergence(std::map<planners::StateSpace, std::vector<uint32_t>>& distributions,
                                                           uint64_t num_states,
                                                           uint64_t num_samples,
                                                           std::chrono::high_resolution_clock::time_point inner_start,
                                                           std::chrono::high_resolution_clock::time_point outer_start,
                                                           uint32_t epoch)
    {
        auto divergence = 0.0;

        for (auto& distribution : distributions)
        {
            auto& state_space = distribution.first;
            auto& distribution_vector = distribution.second;
            const auto uniform_probability = (1.0 / (double) num_states);  // TODO: Rewrite without num_states and remove variable?

            for (const auto& state : state_space->get_states())
            {
                if (state_space->is_goal_state(state))
                {
                    continue;
                }

                if (state_space->is_dead_end_state(state))
                {
                    continue;
                }

                const auto index = state_space->get_unique_index_of_state(state);
                const auto count = (double) distribution_vector[index];

                const auto epsilon = 1E-9;
                const auto probability = count / (double) num_samples;
                divergence += probability * std::log((probability / uniform_probability) + epsilon);
            }

            std::fill(distribution_vector.begin(), distribution_vector.end(), 0);
        }

        log_timings("Divergence", inner_start, outer_start, epoch, divergence);
    }
}  // namespace experiments
