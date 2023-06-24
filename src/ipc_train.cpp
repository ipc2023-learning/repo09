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
#include "private/datasets/fixed_size_dataset.hpp"
#include "private/experiments/reduce_lr_on_plateau.hpp"
#include "private/experiments/value_selfsupervised_suboptimal.hpp"
#include "private/experiments/value_supervised_optimal.hpp"
#include "private/formalism/declarations.hpp"
#include "private/formalism/print_functions.hpp"
#include "private/formalism/problem.hpp"
#include "private/libraries/tclap/CmdLine.h"
#include "private/models/relational_neural_network.hpp"
#include "private/models/utils.hpp"
#include "private/planners/state_space.hpp"
#include "torch/torch.h"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

// Older versions of LibC++ does not have filesystem (e.g., ubuntu 18.04), use the experimental version
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

void handle_sigterm(int signum)
{
    std::cout << "Caught SIGTERM (" << signum << "). Exiting." << std::endl;
    exit(0);
}

struct commandline_parser
{
  private:
    std::string to_string(const std::vector<std::string>& strings) const
    {
        std::string result = "[";

        for (std::size_t i = 0; i < strings.size(); ++i)
        {
            result += strings[i];
            if (i + 1 < strings.size())
            {
                result += ", ";
            }
        }

        result += "]";
        return result;
    }

  public:
    std::string input_path;
    int32_t num_features;
    int32_t num_layers;
    uint32_t batch_size;
    uint32_t chunk_size;
    uint32_t max_epochs;
    int32_t max_state_spaces_size;
    int32_t expand_timeout_s;
    int32_t expand_memory_mb;
    int32_t time_optimal;
    int32_t time_suboptimal;
    double bounds_factor;
    double learning_rate;
    std::string model_path;

    bool parse(int argc, char* argv[], std::string& error_message, int& error_code)
    {
        try
        {
            const int32_t DEFAULT_FEATURES = 64;
            const int32_t DEFAULT_LAYERS = 30;
            const int32_t DEFAULT_BATCHES = 32;
            const int32_t DEFAULT_CHUNKS = 128;
            const int32_t DEFAULT_EPOCHS = 1'000'000;
            const int32_t DEFAULT_MAX_STATE_SPACE_SIZE = 100'000'000;
            const int32_t DEFAULT_TIMEOUT = 300;
            const int32_t DEFAULT_MEMORY = 65536;
            const int32_t DEFAULT_TIME_OPTIMAL = 2100;
            const int32_t DEFAULT_TIME_SUBOPTIMAL = 2100;
            const double DEFAULT_LEARNING_RATE = 0.0002;
            const double DEFAULT_BOUNDS_FACTOR = 2.0;

            // clang-format off
            TCLAP::CmdLine cmd("Muninn", ' ', "IPC 2023: Learning Track");
            TCLAP::ValueArg<std::string> input_arg("", "input", "Path to directory of pddl files (directory)", true, "", "path");
            TCLAP::ValueArg<int32_t> features_arg("", "features", "Number of features per object", false, DEFAULT_FEATURES, "positive integer [default " + std::to_string(DEFAULT_FEATURES) + "]");
            TCLAP::ValueArg<int32_t> layers_arg("", "layers", "Number of layers of the model", false, DEFAULT_LAYERS, "positive integer [default " + std::to_string(DEFAULT_LAYERS) + "]");
            TCLAP::ValueArg<int32_t> batch_arg("", "batch_size", "Maximum size of batches", false, DEFAULT_BATCHES, "positive integer [default " + std::to_string(DEFAULT_BATCHES) + "]");
            TCLAP::ValueArg<int32_t> chunk_arg("", "chunk_size", "Maximum size of chunks", false, DEFAULT_CHUNKS, "positive integer [default " + std::to_string(DEFAULT_CHUNKS) + "]");
            TCLAP::ValueArg<int32_t> epochs_arg("", "epochs", "Number of epochs", false, DEFAULT_EPOCHS, "positive integer [default " + std::to_string(DEFAULT_EPOCHS) + "]");
            TCLAP::ValueArg<double> learning_rate_arg("", "learning_rate", "Learning rate of optimizer", false, DEFAULT_LEARNING_RATE, "positive number [default " + std::to_string(DEFAULT_LEARNING_RATE) + "]");
            TCLAP::ValueArg<double> bounds_factor_arg("", "bounds_factor", "Factor of the bigger bound used to accelerate value function learning", false, DEFAULT_BOUNDS_FACTOR, "negative to disable, positive number [default " + std::to_string(DEFAULT_BOUNDS_FACTOR) + "]");
            TCLAP::ValueArg<int32_t> max_state_space_size_arg("", "max_state_space_size", "Max size of fully expanded state spaces", false, DEFAULT_MAX_STATE_SPACE_SIZE, "positive integer [default " + std::to_string(DEFAULT_MAX_STATE_SPACE_SIZE) + "]");
            TCLAP::ValueArg<int32_t> expand_timeout_arg("", "expand_timeout", "Timeout for expanding each problem (seconds)", false, DEFAULT_TIMEOUT, "positive integer [default " + std::to_string(DEFAULT_TIMEOUT) + "]");
            TCLAP::ValueArg<int32_t> expand_memory_arg("", "expand_memory", "Memory limit for expanding all problems", false, DEFAULT_MEMORY, "positive integer [default " + std::to_string(DEFAULT_MEMORY) + "]");
            TCLAP::ValueArg<int32_t> time_optimal_arg("", "time_optimal", "Time allocated for optimal training (minutes)", false, DEFAULT_TIME_OPTIMAL, "positive integer [default " + std::to_string(DEFAULT_TIME_OPTIMAL) + "]");
            TCLAP::ValueArg<int32_t> time_suboptimal_arg("", "time_suboptimal", "Time allocated for suboptimal training (minutes)", false, DEFAULT_TIME_SUBOPTIMAL, "positive integer [default " + std::to_string(DEFAULT_TIME_SUBOPTIMAL) + "]");
            TCLAP::ValueArg<std::string> load_arg("", "load", "Path to model to resume", false, "", "path");
            // clang-format on

            cmd.add(input_arg);
            cmd.add(features_arg);
            cmd.add(layers_arg);
            cmd.add(batch_arg);
            cmd.add(chunk_arg);
            cmd.add(epochs_arg);
            cmd.add(learning_rate_arg);
            cmd.add(bounds_factor_arg);
            cmd.add(max_state_space_size_arg);
            cmd.add(expand_timeout_arg);
            cmd.add(expand_memory_arg);
            cmd.add(time_optimal_arg);
            cmd.add(time_suboptimal_arg);
            cmd.add(load_arg);

            cmd.parse(argc, argv);

            input_path = input_arg.getValue();
            num_features = features_arg.getValue();
            num_layers = layers_arg.getValue();
            batch_size = batch_arg.getValue();
            chunk_size = chunk_arg.getValue();
            max_epochs = epochs_arg.getValue();
            learning_rate = learning_rate_arg.getValue();
            bounds_factor = bounds_factor_arg.getValue();
            max_state_spaces_size = max_state_space_size_arg.getValue();
            expand_timeout_s = expand_timeout_arg.getValue();
            expand_memory_mb = expand_memory_arg.getValue();
            time_optimal = time_optimal_arg.getValue();
            time_suboptimal = time_suboptimal_arg.getValue();
            model_path = load_arg.getValue();
        }
        catch (TCLAP::ArgException& e)  // catch any exceptions
        {
            error_message = e.error() + "for arg " + e.argId();
            error_code = 1;
            return false;
        }

        if (!fs::exists(input_path))
        {
            error_message = "train path \"" + input_path + "\" does not exist";
            error_code = 2;
            return false;
        }

        error_message = "";
        error_code = 0;
        return true;
    }
};

void train_with_supervised_optimal(const commandline_parser& args,
                                   const std::chrono::high_resolution_clock::time_point& time_start,
                                   const datasets::FixedSizeBalancedDataset& training_set,
                                   const datasets::FixedSizeBalancedDataset& validation_set,
                                   models::RelationalNeuralNetwork& model)
{
    const auto learning_rate_step = [](double learning_rate) { std::cout << "Learning rate: " << learning_rate << std::endl; };
    torch::optim::AdamW optimizer(model.parameters(), args.learning_rate);
    experiments::ReduceLROnPlateau reduce_lr_on_plateau(optimizer, 0.75, 100, args.learning_rate / 10.0);
    reduce_lr_on_plateau.register_on_lr_update(learning_rate_step);

    const auto train_step = [](const experiments::TrainingStep& step, double loss) -> void
    { std::cout << "[" << step.epoch << ", " << step.batch_index + 1 << "/" << step.num_batches << "] Train: " << loss << std::endl; };

    double best_loss = std::numeric_limits<double>::max();
    const auto validation_step = [&time_start, &best_loss, &model, &reduce_lr_on_plateau](const experiments::ValidationStep& step, double loss) -> void
    {
        const auto total_seconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - time_start).count();
        const auto minutes = total_seconds / 60;
        const auto seconds = total_seconds - 60 * minutes;

        std::cout << "[" << step.epoch << "] Validation: " << loss << " (" << minutes << "m " << seconds << "s)" << std::endl;

        if (loss < best_loss)
        {
            best_loss = loss;
            models::save_model("optimal_best", model);
        }

        models::save_model("optimal_latest", model);
        reduce_lr_on_plateau.update(loss);
    };

    const auto time_supervised_end = time_start + std::chrono::minutes(args.time_optimal);
    const auto epoch_step = [&time_supervised_end](const experiments::EpochStep& step) -> bool
    { return std::chrono::high_resolution_clock::now() > time_supervised_end; };

    std::cout << std::endl << "Optimal learning..." << std::endl;
    const auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(std::chrono::high_resolution_clock::now() - time_start).count();
    std::cout << "Time: " << elapsed << " minutes" << std::endl;
    experiments::SupervisedOptimal value_supervised_optimal(args.batch_size, args.chunk_size, args.max_epochs);
    value_supervised_optimal.register_on_training_step(train_step);
    value_supervised_optimal.register_on_validation_step(validation_step);
    value_supervised_optimal.register_on_epoch_step(epoch_step);
    value_supervised_optimal.fit(model, optimizer, training_set, validation_set);
}

void train_with_unsupervised_suboptimal(const commandline_parser& args,
                                        const std::chrono::high_resolution_clock::time_point& time_start,
                                        const datasets::FixedSizeBalancedDataset& training_set,
                                        const datasets::FixedSizeBalancedDataset& validation_set,
                                        models::RelationalNeuralNetwork& model)
{
    const auto learning_rate_step = [](double learning_rate) { std::cout << "Learning rate: " << learning_rate << std::endl; };
    torch::optim::AdamW optimizer(model.parameters(), args.learning_rate);
    experiments::ReduceLROnPlateau reduce_lr_on_plateau(optimizer, 0.75, 100, args.learning_rate / 10.0);
    reduce_lr_on_plateau.register_on_lr_update(learning_rate_step);

    const auto train_step = [](const experiments::TrainingStep& step, double loss) -> void
    { std::cout << "[" << step.epoch << ", " << step.batch_index + 1 << "/" << step.num_batches << "] Train: " << loss << std::endl; };

    double best_loss = std::numeric_limits<double>::max();
    const auto validation_step = [&best_loss, &model, &reduce_lr_on_plateau](const experiments::ValidationStep& step, double loss) -> void
    {
        std::cout << "[" << step.epoch << "] Validation: " << loss << std::endl;

        if (loss < best_loss)
        {
            best_loss = loss;
            models::save_model("suboptimal_best", model);
        }

        models::save_model("suboptimal_latest", model);
        reduce_lr_on_plateau.update(loss);
    };

    const auto time_supervised_end = time_start + std::chrono::minutes(args.time_optimal) + std::chrono::minutes(args.time_suboptimal);
    const auto epoch_step = [&time_supervised_end](const experiments::EpochStep& step) -> bool
    { return std::chrono::high_resolution_clock::now() > time_supervised_end; };

    std::cout << std::endl << "Suboptimal learning..." << std::endl;
    const auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(std::chrono::high_resolution_clock::now() - time_start).count();
    std::cout << "Time: " << elapsed << " minutes" << std::endl;
    experiments::SelfsupervisedSuboptimal value_unsupervised_suboptimal(args.batch_size, args.chunk_size, args.max_epochs, 2.0);
    value_unsupervised_suboptimal.register_on_training_step(train_step);
    value_unsupervised_suboptimal.register_on_validation_step(validation_step);
    value_unsupervised_suboptimal.register_on_epoch_step(epoch_step);
    value_unsupervised_suboptimal.fit(model, optimizer, training_set, validation_set);
}

int main(int argc, char* argv[])
{
    signal(SIGTERM, handle_sigterm);

    const auto time_start = std::chrono::high_resolution_clock::now();

    std::string error_message;
    int error_code;
    commandline_parser args;

    if (!args.parse(argc, argv, error_message, error_code))
    {
        std::cout << "Error: " << error_message << std::endl;
        return error_code;
    }

    const auto problems = load_problems(fs::path(args.input_path));
    const auto predicates = problems.at(0)->domain->predicates;
    const auto types = problems.at(0)->domain->types;

    auto model =
        load_model(args.model_path, "relational_mpnn", predicates, models::DerivedPredicateList(), types, args.num_features, args.num_layers, false, 12.0);
    const auto device = load_device(false);
    model.to(device);

    // Generate training data.

    bool pruning_is_safe = false;
    bool pruning_is_useful = false;
    auto state_spaces =
        compute_state_spaces(problems, args.max_state_spaces_size, true, pruning_is_safe, pruning_is_useful, args.expand_timeout_s, args.expand_memory_mb);

    if (state_spaces.size() == 0)
    {
        std::cerr << "Unable to expand ANY state space. Need at least one instance that can be fully expanded, but preferably five or more." << std::endl;
        std::cerr << "Aborting." << std::endl;
        return 1;
    }

    int use_wl = (pruning_is_safe && pruning_is_useful) ? 1 : 0;
    std::ofstream outputFile("use_wl");
    if (outputFile.is_open())
    {
        outputFile << use_wl;
        outputFile.close();
    }
    else
    {
        std::cerr << "Unable to write \"use_wl\"" << std::endl;
    }

    std::sort(state_spaces.begin(),
              state_spaces.end(),
              [](const planners::StateSpace& lhs, const planners::StateSpace& rhs) -> bool { return lhs->num_states() < rhs->num_states(); });

    const auto training_size = (state_spaces.size() == 1) ? 1 : std::min(state_spaces.size() - 1, (std::size_t)(0.9 * state_spaces.size()));

    planners::StateSpaceList training_state_spaces(state_spaces.begin(), state_spaces.begin() + training_size);
    planners::StateSpaceList validation_state_spaces(state_spaces.begin() + training_size, state_spaces.end());

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

    const datasets::FixedSizeBalancedDataset training_set(training_state_spaces, 10'000);
    const datasets::FixedSizeBalancedDataset validation_set(validation_state_spaces, 1'000);

    train_with_supervised_optimal(args, time_start, training_set, validation_set, model);
    train_with_unsupervised_suboptimal(args, time_start, training_set, validation_set, model);

    const auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(std::chrono::high_resolution_clock::now() - time_start).count();
    std::cout << "Time: " << elapsed << " minutes" << std::endl;

    return 0;
}
