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


#include "helpers.hpp"
#include "private/datasets/balanced_dataset.hpp"
#include "private/datasets/dataset.hpp"
#include "private/datasets/fixed_size_dataset.hpp"
#include "private/datasets/random_dataset.hpp"
#include "private/experiments/dataset_experiment.hpp"
#include "private/experiments/value_selfsupervised_suboptimal.hpp"
#include "private/experiments/value_supervised_optimal.hpp"
#include "private/formalism/declarations.hpp"
#include "private/formalism/problem.hpp"
#include "private/models/utils.hpp"
#include "private/planners/state_space.hpp"
#include "train_dl.hpp"

#include <chrono>
#include <iostream>

std::vector<std::string> dataset_method_types() { return std::vector<std::string>({ "value_supervised_optimal", "value_selfsupervised_suboptimal" }); }

double mean_unit_cost_to_goal(const std::shared_ptr<datasets::Dataset>& dataset)
{
    uint32_t total = 0;

    for (uint32_t index = 0; index < dataset->size(); ++index)
    {
        const auto sample = dataset->get(index);
        const auto& state = sample.first;
        const auto& state_space = sample.second;
        total += state_space->get_distance_to_goal_state(state);
    }

    return (double) total / (double) dataset->size();
}

double variance_unit_cost_to_goal(const std::shared_ptr<datasets::Dataset>& dataset)
{
    const auto mean = mean_unit_cost_to_goal(dataset);
    uint32_t var = 0;

    for (uint32_t index = 0; index < dataset->size(); ++index)
    {
        const auto sample = dataset->get(index);
        const auto& state = sample.first;
        const auto& state_space = sample.second;
        const auto deviation = (state_space->get_distance_to_goal_state(state) - mean);
        var += deviation * deviation;
    }

    return std::sqrt((double) var / (double) dataset->size());
}

void train_dl(const DatasetLearningSettings& settings, const formalism::ProblemDescriptionList& problems, models::RelationalNeuralNetwork& model)
{
    planners::StateSpaceList small_state_spaces = compute_state_spaces(problems, 100'000, settings.use_weisfeiler_leman);

    std::sort(small_state_spaces.begin(), small_state_spaces.end(), [](const planners::StateSpace& lhs, const planners::StateSpace& rhs) -> bool {
        return lhs->num_states() < rhs->num_states();
    });

    const auto training_size = (small_state_spaces.size() == 1) ? 1 : std::min(small_state_spaces.size() - 1, (std::size_t)(0.8 * small_state_spaces.size()));

    planners::StateSpaceList training_state_spaces(small_state_spaces.begin(), small_state_spaces.begin() + training_size);
    planners::StateSpaceList validation_state_spaces(small_state_spaces.begin() + training_size, small_state_spaces.end());

    std::cout << "Training problems: " << std::endl;
    for (const auto& state_space : training_state_spaces)
    {
        std::cout << " - " << state_space->problem->name << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Validation problems: " << std::endl;
    for (const auto& state_space : validation_state_spaces)
    {
        std::cout << " - " << state_space->problem->name << std::endl;
    }
    std::cout << std::endl;

    std::shared_ptr<datasets::Dataset> training_set = nullptr;
    std::shared_ptr<datasets::Dataset> validation_set = nullptr;

    if (settings.disable_balancing)
    {
        training_set = std::make_shared<datasets::RandomDataset>(training_state_spaces, false);
        validation_set = std::make_shared<datasets::RandomDataset>(validation_state_spaces, false);
    }
    else
    {
        training_set = std::make_shared<datasets::FixedSizeBalancedDataset>(training_state_spaces, 10'000);
        validation_set = std::make_shared<datasets::FixedSizeBalancedDataset>(validation_state_spaces, 1'000);
    }

    // Callback functions for experiment

    const auto time_start = std::chrono::high_resolution_clock::now();

    const auto train_step = [](const experiments::TrainingStep& step, double loss) -> void
    { std::cout << "[" << step.epoch << ", " << step.batch_index + 1 << "/" << step.num_batches << "] Train: " << loss << std::endl; };

    double best_loss = std::numeric_limits<double>::max();
    const auto validation_step = [&best_loss, &model, &time_start](const experiments::ValidationStep& step, double loss) -> void
    {
        const auto time_now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(time_now - time_start).count();

        // Convert the duration to hours, minutes, and seconds
        const auto hours = duration / 3600;
        duration -= hours * 3600;
        const auto minutes = duration / 60;
        duration -= minutes * 60;
        const auto seconds = duration;

        std::cout << "[" << step.epoch << "] Validation: " << loss;
        std::cout << " (";
        std::cout << std::setfill('0') << std::setw(2) << hours << ":";
        std::cout << std::setfill('0') << std::setw(2) << minutes << ":";
        std::cout << std::setfill('0') << std::setw(2) << seconds;
        std::cout << ")" << std::endl;

        if (loss < best_loss)
        {
            best_loss = loss;
            models::save_model("best", model);
        }

        models::save_model("latest", model);
    };

    try
    {
        std::shared_ptr<experiments::DatasetExperiment> experiment(nullptr);

        if (settings.method == "value_supervised_optimal")
        {
            experiment = std::make_shared<experiments::SupervisedOptimal>(settings.batch_size, settings.chunk_size, settings.max_epochs);
        }
        else if (settings.method == "value_selfsupervised_suboptimal")
        {
            experiment =
                std::make_shared<experiments::SelfsupervisedSuboptimal>(settings.batch_size, settings.chunk_size, settings.max_epochs, settings.bounds_factor);
        }
        else
        {
            throw std::invalid_argument(settings.method + " is not implemented");
        }

        torch::optim::AdamW optimizer(model.parameters(), settings.learning_rate);
        experiment->register_on_training_step(train_step);
        experiment->register_on_validation_step(validation_step);
        experiment->fit(model, optimizer, *training_set, *validation_set);
    }
    catch (const std::exception& ex)
    {
        std::cout << "error: " << ex.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "unknown error occurred" << std::endl;
    }
}
