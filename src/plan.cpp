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


#include "private/datasets/dataset.hpp"
#include "private/formalism/print_functions.hpp"
#include "private/libraries/tclap/CmdLine.h"
#include "private/models/relational_neural_network.hpp"
#include "private/models/utils.hpp"
#include "private/pddl/domain_parser.hpp"
#include "private/pddl/problem_parser.hpp"
#include "private/planners/generators/applicable_actions_naive_generator.hpp"
#include "private/planners/policy_search.hpp"
#include "private/planners/value_search.hpp"
#include "torch/torch.h"

#include <chrono>
#include <iomanip>

// Older versions of LibC++ does not have filesystem (e.g., ubuntu 18.04), use the experimental version
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

template<typename InputType, typename ReturnType>
void print_vector(std::vector<InputType> vector, std::function<ReturnType(InputType)> func, std::string delimiter)
{
    for (uint32_t index = 0; index < vector.size(); ++index)
    {
        std::cout << func(vector.at(index));

        if (index + 1 < vector.size())
        {
            std::cout << delimiter;
        }
    }
}

void print_domain(const formalism::DomainDescription& domain)
{
    std::cout << "Domain: " << domain->name << std::endl;

    std::cout << "Requirements: ";
    print_vector<std::string, std::string>(
        domain->requirements,
        [](auto requirement) { return requirement; },
        ", ");
    std::cout << std::endl;

    std::cout << "Types: ";
    print_vector<formalism::Type, std::string>(
        domain->types,
        [](auto type) { return type->name; },
        ", ");
    std::cout << std::endl;

    std::cout << "Constants: ";
    print_vector<formalism::Object, std::string>(
        domain->constants,
        [](auto constant) { return constant->name; },
        ", ");
    std::cout << std::endl;

    std::cout << "Predicates: ";
    print_vector<formalism::Predicate, std::string>(
        domain->predicates,
        [](auto predicate) { return predicate->name + "/" + std::to_string(predicate->arity); },
        ", ");
    std::cout << std::endl;

    std::cout << "Action schemas:" << std::endl;
    for (const auto& action_schema : domain->action_schemas)
    {
        std::cout << action_schema->name << "(";
        print_vector<formalism::Parameter, std::string>(
            action_schema->parameters,
            [](auto parameter) { return parameter->name + ": " + parameter->type->name; },
            ", ");
        std::cout << ")" << std::endl;

        if (action_schema->complete)
        {
            std::cout << "pre: " << std::endl;
            for (const auto& literal : action_schema->precondition)
            {
                std::cout << " " << (literal->negated ? "-" : "+") << literal->atom->predicate->name << "(";
                print_vector<formalism::Object, std::string>(
                    literal->atom->arguments,
                    [](auto argument) { return argument->name; },
                    ", ");
                std::cout << ")" << std::endl;
            }

            std::cout << "eff: " << std::endl;
            for (const auto& literal : action_schema->effect)
            {
                std::cout << " " << (literal->negated ? "-" : "+") << literal->atom->predicate->name << "(";
                print_vector<formalism::Object, std::string>(
                    literal->atom->arguments,
                    [](auto argument) { return argument->name; },
                    ", ");
                std::cout << ")" << std::endl;
            }
        }

        std::cout << std::endl;
    }
}

torch::Device load_device(bool force_cpu)
{
    if (torch::cuda::is_available() && !force_cpu)
    {
        std::cout << "GPU/CUDA device is available and will be used" << std::endl << std::endl;
        return torch::kCUDA;
    }
    else if (torch::cuda::is_available() && force_cpu)
    {
        std::cout << "GPU/CUDA device is available but will NOT be used" << std::endl << std::endl;
        return torch::kCPU;
    }
    else
    {
        std::cout << "Only CPU is available and will be used" << std::endl << std::endl;
        return torch::kCPU;
    }
}

int main(int argc, char* argv[])
{
    std::string domain_path;
    std::string problem_path;
    std::string model_path;
    std::string method_type;
    bool closed;
    bool cpu;
    bool parameters;
    bool verbose;
    bool deterministic_policy;

    try
    {
        // clang-format off
        TCLAP::CmdLine cmd("", ' ', "0.1");
        TCLAP::ValueArg<std::string> domain_arg("", "domain", "Path to domain file (if not given, file \"domain.pddl\" will be assumed to be in the directory as the given problem)", false, "", "path");
        TCLAP::ValueArg<std::string> problem_arg("", "problem", "Path to problem file", true, "", "path");
        TCLAP::ValueArg<std::string> model_arg("", "model", "Path to model", true, "", "path");
        TCLAP::ValueArg<std::string> method_arg("", "method", "One of [policy, value]", true, "", "type");
        TCLAP::SwitchArg closed_arg("", "closed", "Use closed set", false);
        TCLAP::SwitchArg deterministic_policy_arg("", "deterministic_policy", "Always take the most probable policy transition", false);
        TCLAP::SwitchArg cpu_arg("", "cpu", "Use CPU even if GPU is available", false);
        TCLAP::SwitchArg parameters_arg("", "parameters", "Print parameter distribution of the neural network", false);
        TCLAP::SwitchArg verbose_arg("", "verbose", "Print detailed planning information", false);
        // clang-format on

        cmd.add(domain_arg);
        cmd.add(problem_arg);
        cmd.add(model_arg);
        cmd.add(method_arg);
        cmd.add(closed_arg);
        cmd.add(deterministic_policy_arg);
        cmd.add(cpu_arg);
        cmd.add(parameters_arg);
        cmd.add(verbose_arg);

        cmd.parse(argc, argv);

        domain_path = domain_arg.getValue();
        problem_path = problem_arg.getValue();
        model_path = model_arg.getValue();
        method_type = method_arg.getValue();

        closed = closed_arg.getValue();
        deterministic_policy = deterministic_policy_arg.getValue();
        cpu = cpu_arg.getValue();
        parameters = parameters_arg.getValue();
        verbose = verbose_arg.getValue();

        // Normalize representations

        std::transform(method_type.begin(), method_type.end(), method_type.begin(), [](unsigned char c) { return std::tolower(c); });
    }
    catch (TCLAP::ArgException& e)  // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        return 1;
    }

    if (!fs::exists(problem_path))
    {
        std::cout << "Error: \"problem\" does not exist" << std::endl;
        return 2;
    }

    if (domain_path == "")
    {
        domain_path = (fs::path(problem_path).parent_path() / "domain.pddl").string();
    }

    if (!fs::exists(domain_path))
    {
        std::cout << "Error: \"domain\" does not exist" << std::endl;
        return 3;
    }

    // parse PDDL files

    parsers::DomainParser domain_parser(domain_path);
    const auto domain = domain_parser.parse();
    std::cout << domain << std::endl;

    parsers::ProblemParser problem_parser(problem_path);
    const auto problem = problem_parser.parse(domain);
    std::cout << problem << std::endl;

    // load model

    models::RelationalNeuralNetwork model;
    if (models::load_model(&model, model_path))
    {
        // TODO: Generalize...

        std::cout << "Loaded model from file: " << std::endl;
        std::cout << model;
    }
    else
    {
        std::cout << "Error: could not load model" << std::endl;
        return 4;
    }

    auto device = load_device(cpu);
    model.to(device);
    model.eval();
    torch::NoGradGuard no_grad;

    // print parameter distribution

    if (parameters)
    {
        std::cout << "Parameter Distribution:" << std::endl;
        std::unordered_map<int32_t, int32_t> counts;
        const double DISCRETIZATION_FACTOR = 100.0;
        for (const auto& parameter_list : model.parameters())
        {
            const auto num_parameters = parameter_list.numel();
            for (int64_t parameter_index = 0; parameter_index < num_parameters; ++parameter_index)
            {
                const auto parameter = parameter_list.view(-1)[parameter_index];
                const auto index = (int32_t)(DISCRETIZATION_FACTOR * parameter.item<double>());
                counts[index] += 1;
            }
        }
        std::vector<std::pair<int32_t, int32_t>> histogram(counts.begin(), counts.end());
        std::sort(histogram.begin(), histogram.end());

        for (const auto& entry : histogram)
        {
            std::cout << "  " << (entry.first / DISCRETIZATION_FACTOR) << ": " << entry.second << std::endl;
        }
        std::cout << std::endl;
    }
    // find  plan

    std::cout << "Begin execution of policy..." << std::endl;
    std::vector<formalism::Action> plan;
    bool found_plan = false;

    if (method_type == "policy")
    {
        planners::PolicySearch planner(problem, model);
        found_plan = planner.find_plan(verbose, deterministic_policy, closed, plan);

        // TODO: Inherit planners::SearchBase or something.

        std::cout << "Expanded " << planner.expanded << " states" << std::endl;
        std::cout << "Generated " << planner.generated << " states" << std::endl << std::endl;
        std::cout << "Successor time: " << planner.time_successors_ns / (int64_t) 1E6 << " ms"
                  << " (" << std::fixed << std::setprecision(3) << (100.0 * planner.time_successors_ns) / planner.time_total_ns << "%)" << std::endl;
        std::cout << "Inference time: " << planner.time_inference_ns / (int64_t) 1E6 << " ms"
                  << " (" << std::fixed << std::setprecision(3) << (100.0 * planner.time_inference_ns) / planner.time_total_ns << "%)" << std::endl;
        std::cout << "Total time: " << planner.time_total_ns / (int64_t) 1E6 << " ms" << std::endl;
    }
    else if (method_type == "value")
    {
        planners::ValueSearch planner(problem, model);
        found_plan = planner.find_plan(verbose, plan);

        std::cout << "Expanded " << planner.expanded << " states" << std::endl;
        std::cout << "Generated " << planner.generated << " states" << std::endl << std::endl;
        std::cout << "Successor time: " << planner.time_successors_ns / (int64_t) 1E6 << " ms"
                  << " (" << std::fixed << std::setprecision(3) << (100.0 * planner.time_successors_ns) / planner.time_total_ns << "%)" << std::endl;
        std::cout << "Inference time: " << planner.time_inference_ns / (int64_t) 1E6 << " ms"
                  << " (" << std::fixed << std::setprecision(3) << (100.0 * planner.time_inference_ns) / planner.time_total_ns << "%)" << std::endl;
        std::cout << "Total time: " << planner.time_total_ns / (int64_t) 1E6 << " ms" << std::endl;
    }
    else
    {
        std::cout << "invalid method" << std::endl;
        return 5;
    }

    if (found_plan)
    {
        std::cout << "Plan: (" << plan.size() << " actions)" << std::endl;
        for (const auto& action : plan)
        {
            std::cout << action << std::endl;
        }
    }
    else
    {
        std::cout << "Failed to find a plan (stopped after " << plan.size() << " actions)" << std::endl;
    }

    return 0;
}
