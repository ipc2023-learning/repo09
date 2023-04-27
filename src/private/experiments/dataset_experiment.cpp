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
#include "dataset_experiment.hpp"
#include "reduce_lr_on_plateau.hpp"

#include <chrono>

namespace experiments
{
    std::pair<torch::Tensor, torch::Tensor>
    forward_in_chunks(models::RelationalNeuralNetwork& model, const planners::StateSpaceSampleList& batch, uint32_t chunk_size)
    {
        // Some loss functions include successors or predecessors in the batch, i.e. the batch might be larger than batch_size.
        // In order to make memory consumption more predictable, process the batch in (up to) batch_size large pieces.

        const uint32_t batch_size = (uint32_t) batch.size();
        const uint32_t num_chunks = (batch_size / chunk_size) + ((batch_size % chunk_size) > 0 ? 1 : 0);

        auto values = torch::zeros({ 0, 1 }).to(model.device());
        auto dead_ends = torch::zeros({ 0, 1 }).to(model.device());

        for (uint32_t chunk_index = 0; chunk_index < num_chunks; ++chunk_index)
        {
            auto first = batch.begin() + chunk_index * chunk_size;
            auto last = batch.begin() + std::min(batch_size, chunk_index * chunk_size + chunk_size);
            formalism::StateProblemList chunk;

            while (first != last)
            {
                const auto& sample = *first;
                chunk.push_back(std::make_pair(sample.first, sample.second->problem));
                ++first;
            }

            const auto [output_values, output_dead_ends] = model.forward(chunk);
            values = torch::cat({ values, output_values }, 0);
            dead_ends = torch::cat({ dead_ends, output_dead_ends }, 0);
        }

        return std::make_pair(values, dead_ends);
    }

    void DatasetExperiment::register_on_training_step(const OnTrainingStepCallback& callback) { training_callback_ = callback; }

    void DatasetExperiment::register_on_validation_step(const OnValidationStepCallback& callback) { validation_callback_ = callback; }

    void DatasetExperiment::register_on_epoch_step(const OnEpochStepCallback& callback) { epoch_callback_ = callback; }

    planners::StateSpaceSampleList DatasetExperiment::get_batch(const datasets::Dataset& set, uint32_t batch_index, uint32_t batch_size)
    {
        return set.get_range(batch_index * batch_size, batch_size);
    }

    void DatasetExperiment::fit(models::RelationalNeuralNetwork& model,
                                torch::optim::Optimizer& optimizer,
                                const datasets::Dataset& training_set,
                                const datasets::Dataset& validation_set)
    {
        const uint32_t num_train_batches = (training_set.size() / batch_size) + ((training_set.size() % batch_size) > 0 ? 1 : 0);
        const uint32_t num_val_batches = (validation_set.size() / batch_size) + ((validation_set.size() % batch_size) > 0 ? 1 : 0);

        const auto time_start = std::chrono::high_resolution_clock::now();

        for (uint32_t epoch = 0; epoch < max_epochs; epoch++)
        {
            model.train();
            for (uint32_t batch_index = 0; batch_index < num_train_batches; ++batch_index)
            {
                model.zero_grad();
                const auto batch = get_batch(training_set, batch_index, batch_size);
                const auto output = forward_in_chunks(model, batch, chunk_size);
                const auto loss = train_loss(batch, output);
                loss.backward();
                optimizer.step();

                if (training_callback_)
                {
                    const auto time_now = std::chrono::high_resolution_clock::now();
                    const auto time_ms = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(time_now - time_start).count());
                    const auto training_loss = loss.item<double>();
                    training_callback_({ .epoch = epoch, .batch_index = batch_index, .num_batches = num_train_batches, .time_ms = time_ms }, training_loss);
                }
            }

            if (validation_callback_)
            {
                // Disable computation of gradients to speed up the computation.
                torch::NoGradGuard no_grad;
                model.eval();
                auto total_validation_loss = 0.0;  // It would be better to do this on the device, but that crashes for some reason
                for (uint32_t batch_index = 0; batch_index < num_val_batches; ++batch_index)
                {
                    const auto batch = get_batch(validation_set, batch_index, batch_size);
                    const auto output = forward_in_chunks(model, batch, chunk_size);
                    const auto loss = validation_loss(batch, output);
                    total_validation_loss += batch.size() * loss.item<double>();
                }

                const auto time_now = std::chrono::high_resolution_clock::now();
                const auto time_ms = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(time_now - time_start).count());
                const auto validation_loss = total_validation_loss / (double) validation_set.size();
                validation_callback_({ .epoch = epoch, .time_ms = time_ms }, validation_loss);
            }

            if (epoch_callback_)
            {
                const auto time_now = std::chrono::high_resolution_clock::now();
                const auto time_ms = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(time_now - time_start).count());
                const auto abort = epoch_callback_({ .epoch = epoch, .time_ms = time_ms });

                if (abort)
                {
                    break;
                }
            }
        }
    }
}  // namespace experiments
