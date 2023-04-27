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
#include "private/experiments/policy_evaluation.hpp"
#include "private/formalism/declarations.hpp"
#include "private/formalism/print_functions.hpp"
#include "private/libraries/json.hpp"
#include "private/libraries/tclap/CmdLine.h"
#include "private/models/relational_neural_network.hpp"
#include "torch/torch.h"
#include "train_dl.hpp"
#include "train_ol.hpp"
#include "train_rl.hpp"

#include <algorithm>
#include <chrono>
#include <cwctype>

// Older versions of LibC++ does not have filesystem (e.g., ubuntu 18.04), use the experimental version
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

std::vector<std::string> model_types() { return std::vector<std::string>({ "relational_mpnn", "relational_transformer" }); }

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
    std::string method_type;
    std::string model_type;
    std::string reward_type;
    std::string derived_predicates_path;
    int32_t num_features;
    int32_t num_layers;
    int32_t horizon;
    double maximum_smoothness;
    uint32_t batch_size;
    uint32_t chunk_size;
    uint32_t max_epochs;
    double discount;
    double bounds_factor;
    double learning_rate;
    bool global_readout;
    bool disable_balancing;
    bool disable_baseline;
    bool disable_value_regularization;
    bool use_weisfeiler_leman;
    bool cpu;
    std::string model_path;

    bool parse(int argc, char* argv[], std::string& error_message, int& error_code)
    {
        const auto rl_methods = reinforcement_learning_method_types();
        const auto dl_methods = dataset_method_types();
        const auto ol_methods = other_method_types();

        std::vector<std::string> all_method_types;
        all_method_types.insert(all_method_types.end(), rl_methods.begin(), rl_methods.end());
        all_method_types.insert(all_method_types.end(), dl_methods.begin(), dl_methods.end());
        all_method_types.insert(all_method_types.end(), ol_methods.begin(), ol_methods.end());

        const auto models = model_types();

        try
        {
            const int32_t DEFAULT_FEATURES = 64;
            const int32_t DEFAULT_LAYERS = 30;
            const int32_t DEFAULT_HORIZON = 50;
            const double DEFAULT_MAXIMUM_SMOOTHNESS = 12.0;
            const int32_t DEFAULT_BATCHES = 64;
            const int32_t DEFAULT_CHUNKS = 128;
            const int32_t DEFAULT_EPOCHS = 10000;
            const double DEFAULT_LEARNING_RATE = 0.0002;
            const double DEFAULT_DISCOUNT = 0.999;
            const double DEFAULT_BOUNDS_FACTOR = 2.0;
            const std::string DEFAULT_REWARD = "blind";
            const std::string DEFAULT_DERIVED_PREDICATES = "";

            // clang-format off
            TCLAP::CmdLine cmd("", ' ', "0.1");
            TCLAP::ValueArg<std::string> input_arg("", "input", "Path to directory of pddl files (directory)", true, "", "path");
            TCLAP::ValueArg<std::string> method_arg("", "method", "One of " + to_string(all_method_types), true, "", "type");
            TCLAP::ValueArg<std::string> model_arg("", "model", "One of " + to_string(models), false, "relational_mpnn", "type");
            TCLAP::ValueArg<std::string> reward_arg("", "reward", "One of " + to_string(reinforcement_learning_reward_types()) + " (only for RL methods)", false, DEFAULT_REWARD, "type [default " + DEFAULT_REWARD + "]");
            TCLAP::ValueArg<int32_t> features_arg("", "features", "Number of features per object", false, DEFAULT_FEATURES, "positive integer [default " + std::to_string(DEFAULT_FEATURES) + "]");
            TCLAP::ValueArg<int32_t> layers_arg("", "layers", "Number of layers of the model", false, DEFAULT_LAYERS, "positive integer [default " + std::to_string(DEFAULT_LAYERS) + "]");
            TCLAP::ValueArg<std::string> derived_predicates_arg("", "derived_predicates", "Path to file with definitions of derived predicates", false, DEFAULT_DERIVED_PREDICATES, "path");
            TCLAP::ValueArg<int32_t> horizon_arg("", "horizon", "Max number of steps in online learning", false, DEFAULT_HORIZON, "positive integer [default " + std::to_string(DEFAULT_HORIZON) + "]");
            TCLAP::ValueArg<double> maximum_smoothness_arg("", "maximum_smoothness", "Smoothness of maximum aggregation", false, DEFAULT_MAXIMUM_SMOOTHNESS, "positive number [default " + std::to_string(DEFAULT_MAXIMUM_SMOOTHNESS) + "]");
            TCLAP::ValueArg<int32_t> batches_arg("", "batch_size", "Maximum size of batches", false, DEFAULT_BATCHES, "positive integer [default " + std::to_string(DEFAULT_BATCHES) + "]");
            TCLAP::ValueArg<int32_t> chunk_arg("", "chunk_size", "Maximum size of chunks", false, DEFAULT_CHUNKS, "positive integer [default " + std::to_string(DEFAULT_CHUNKS) + "]");
            TCLAP::ValueArg<int32_t> epochs_arg("", "epochs", "Number of epochs", false, DEFAULT_EPOCHS, "positive integer [default " + std::to_string(DEFAULT_EPOCHS) + "]");
            TCLAP::ValueArg<double> learning_rate_arg("", "learning_rate", "Learning rate of optimizer", false, DEFAULT_LEARNING_RATE, "positive number [default " + std::to_string(DEFAULT_LEARNING_RATE) + "]");
            TCLAP::ValueArg<double> discount_arg("", "discount", "Non-goal reward size", false, DEFAULT_DISCOUNT, "number between 0 and 1 [default " + std::to_string(DEFAULT_DISCOUNT) + "]");
            TCLAP::ValueArg<double> bounds_factor_arg("", "bounds_factor", "Factor of the bigger bound used to accelerate value function learning", false, DEFAULT_BOUNDS_FACTOR, "negative to disable, positive number [default " + std::to_string(DEFAULT_BOUNDS_FACTOR) + "]");
            TCLAP::ValueArg<std::string> load_arg("", "load", "Path to model to resume", false, "", "path");
            TCLAP::SwitchArg global_readout_arg("", "global_readout", "Use global readout in the forward pass", false);
            TCLAP::SwitchArg disable_balancing_arg("", "disable_balancing", "Do not balance datasets, if relevant", false);
            TCLAP::SwitchArg disable_baseline_arg("", "disable_baseline", "Do not use baseline, if relevant", false);
            TCLAP::SwitchArg disable_value_regularization_arg("", "disable_value_regularization", "Do not train value function, if relevant for policies", false);
            TCLAP::SwitchArg use_weisfeiler_leman_arg("", "use_weisfeiler_leman", "Use weisfeiler_leman when expanding state-spaces", false);
            TCLAP::SwitchArg cpu_arg("c", "cpu", "Use CPU even if GPU is available", false);
            // clang-format on

            cmd.add(input_arg);
            cmd.add(method_arg);
            cmd.add(model_arg);
            cmd.add(reward_arg);
            cmd.add(features_arg);
            cmd.add(layers_arg);
            cmd.add(derived_predicates_arg);
            cmd.add(horizon_arg);
            cmd.add(maximum_smoothness_arg);
            cmd.add(batches_arg);
            cmd.add(chunk_arg);
            cmd.add(epochs_arg);
            cmd.add(learning_rate_arg);
            cmd.add(discount_arg);
            cmd.add(bounds_factor_arg);
            cmd.add(load_arg);
            cmd.add(global_readout_arg);
            cmd.add(disable_balancing_arg);
            cmd.add(disable_baseline_arg);
            cmd.add(disable_value_regularization_arg);
            cmd.add(use_weisfeiler_leman_arg);
            cmd.add(cpu_arg);

            cmd.parse(argc, argv);

            input_path = input_arg.getValue();
            method_type = method_arg.getValue();
            model_type = model_arg.getValue();
            reward_type = reward_arg.getValue();
            derived_predicates_path = derived_predicates_arg.getValue();
            num_features = features_arg.getValue();
            num_layers = layers_arg.getValue();
            horizon = horizon_arg.getValue();
            maximum_smoothness = maximum_smoothness_arg.getValue();
            batch_size = batches_arg.getValue();
            chunk_size = chunk_arg.getValue();
            max_epochs = epochs_arg.getValue();
            learning_rate = learning_rate_arg.getValue();
            discount = discount_arg.getValue();
            bounds_factor = bounds_factor_arg.getValue();
            model_path = load_arg.getValue();
            global_readout = global_readout_arg.getValue();
            disable_balancing = disable_balancing_arg.getValue();
            disable_baseline = disable_baseline_arg.getValue();
            disable_value_regularization = disable_value_regularization_arg.getValue();
            use_weisfeiler_leman = use_weisfeiler_leman_arg.getValue();
            cpu = cpu_arg.getValue();

            // Normalize representations

            std::transform(method_type.begin(), method_type.end(), method_type.begin(), [](unsigned char c) { return std::tolower(c); });
            std::transform(reward_type.begin(), reward_type.end(), reward_type.begin(), [](unsigned char c) { return std::tolower(c); });
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

        if ((derived_predicates_path.size() > 0) && !fs::exists(derived_predicates_path))
        {
            error_message = "derived predicates path \"" + derived_predicates_path + "\" does not exist";
            error_code = 3;
            return false;
        }

        if (!contains(all_method_types, method_type))
        {
            error_message = "method \"" + method_type + "\" does not exist";
            error_code = 4;
            return false;
        }

        if (!contains(models, model_type))
        {
            error_message = "model \"" + model_type + "\" does not exist";
            error_code = 5;
            return false;
        }

        if (!contains(reinforcement_learning_reward_types(), reward_type))
        {
            error_message = "reward \"" + reward_type + "\" does not exist";
            error_code = 6;
            return false;
        }

        error_message = "";
        error_code = 0;
        return true;
    }
};

models::DerivedPredicateParams parse_atom(const std::string& atom)
{
    auto simplified_atom = atom;
    simplified_atom.erase(std::remove_if(simplified_atom.begin(), simplified_atom.end(), std::iswspace), simplified_atom.end());
    const auto param_start_index = simplified_atom.find('(');
    const auto param_end_index = simplified_atom.find(')', param_start_index);
    const auto name = simplified_atom.substr(0, param_start_index);

    std::vector<std::string> params;
    std::size_t current_delimiter_index = param_start_index + 1;

    while (true)
    {
        const auto next_delimiter_index = simplified_atom.find(',', current_delimiter_index);

        if (next_delimiter_index == simplified_atom.npos)
        {
            if (current_delimiter_index < param_end_index)
            {
                params.push_back(simplified_atom.substr(current_delimiter_index, param_end_index - current_delimiter_index));
            }

            break;
        }
        else
        {
            params.push_back(simplified_atom.substr(current_delimiter_index, next_delimiter_index - current_delimiter_index));
        }

        current_delimiter_index = next_delimiter_index + 1;
    }

    return std::make_pair(name, params);
}

int main(int argc, char* argv[])
{
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
    models::DerivedPredicateList derived_predicates;

    if (fs::exists(args.derived_predicates_path))
    {
        std::ifstream derived_predicates_stream(args.derived_predicates_path);
        nlohmann::json derived_predicates_json = nlohmann::json::parse(derived_predicates_stream);
        const auto derived_predicates_parsed = derived_predicates_json.get<std::map<std::string, std::vector<std::vector<std::string>>>>();

        for (const auto& kvp : derived_predicates_parsed)
        {
            const auto derived_predicate_params = parse_atom(kvp.first);
            models::DerivedPredicateCaseList derived_predicate_case_list;

            for (const auto& case_list : kvp.second)
            {
                models::DerivedPredicateCase derived_predicate_case;

                for (const auto& case_atom : case_list)
                {
                    derived_predicate_case.push_back(parse_atom(case_atom));
                }

                derived_predicate_case_list.push_back(std::move(derived_predicate_case));
            }

            const auto derived_predicate = std::make_pair(derived_predicate_params, derived_predicate_case_list);
            derived_predicates.push_back(std::move(derived_predicate));
        }
    }

    auto model = load_model(args.model_path,
                            args.model_type,
                            predicates,
                            derived_predicates,
                            types,
                            args.num_features,
                            args.num_layers,
                            args.global_readout,
                            args.maximum_smoothness);
    const auto device = load_device(args.cpu);
    model.to(device);

    if (contains(dataset_method_types(), args.method_type))
    {
        DatasetLearningSettings settings;
        settings.method = args.method_type;
        settings.batch_size = args.batch_size;
        settings.chunk_size = args.chunk_size;
        settings.max_epochs = args.max_epochs;
        settings.bounds_factor = args.bounds_factor;
        settings.learning_rate = args.learning_rate;
        settings.disable_balancing = args.disable_balancing;
        settings.use_weisfeiler_leman = args.use_weisfeiler_leman;

        train_dl(settings, problems, model);
    }
    else if (contains(reinforcement_learning_method_types(), args.method_type))
    {
        ReinforcementLearningSettings settings;
        settings.method = args.method_type;
        settings.reward = args.reward_type;
        settings.batch_size = args.batch_size;
        settings.max_epochs = args.max_epochs;
        settings.horizon = args.horizon;
        settings.learning_rate = args.learning_rate;
        settings.discount_factor = args.discount;
        settings.bounds_factor = args.bounds_factor;
        settings.use_weisfeiler_leman = args.use_weisfeiler_leman;

        train_rl(settings, problems, model);
    }
    else if (contains(other_method_types(), args.method_type))
    {
        OtherLearningSettings settings;
        settings.method = args.method_type;
        settings.batch_size = args.batch_size;
        settings.chunk_size = args.chunk_size;
        settings.max_epochs = args.max_epochs;
        settings.trajectory_length = (uint32_t) std::max(1, args.horizon);
        settings.horizon = args.horizon;
        settings.learning_rate = args.learning_rate;
        settings.discount_factor = args.discount;
        settings.disable_balancing = args.disable_balancing;
        settings.disable_baseline = args.disable_baseline;
        settings.disable_value_regularization = args.disable_value_regularization;
        settings.use_weisfeiler_leman = args.use_weisfeiler_leman;

        train_ol(settings, problems, model);
    }
    else
    {
        std::cout << "[Internal Error] train.cpp: This should have been caught by argument validation" << std::endl;
        return 101;
    }

    return 0;
}
