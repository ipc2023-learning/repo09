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
#include "private/planners/batched_astar_search.hpp"
#include "private/planners/batched_bfs_search.hpp"
#include "private/planners/conditions/always_condition.hpp"
#include "private/planners/conditions/goal_condition.hpp"
#include "private/planners/conditions/one_novelty_condition.hpp"
#include "private/planners/conditions/two_novelty_condition.hpp"
#include "private/planners/conditions/weisfeiler_leman_condition.hpp"
#include "private/planners/generators/successor_generator_factory.hpp"
#include "private/planners/heuristics/blind_heuristic.hpp"
#include "private/planners/heuristics/relational_neural_network_heuristic.hpp"
#include "private/planners/heuristics/scale_heuristic.hpp"
#include "private/planners/search_base.hpp"
#include "private/planners/value_decomposition_search.hpp"
#include "torch/torch.h"

#include <chrono>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>

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
    else if (torch::cuda::is_available() && !force_cpu)
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

void write_plan(const std::string& filename, const std::vector<formalism::Action>& plan)
{
    std::cout << "Writing file \"" << filename << "\" (" << plan.size() << " actions)" << std::endl;

    std::ofstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (const auto& action : plan)
    {
        file << "(";
        file << action->schema->name;

        for (const auto& argument : action->arguments)
        {
            file << " ";
            file << argument;
        }
        file << ")";
        file << std::endl;
    }

    file << std::endl;
}

int main(int argc, char* argv[])
{
    signal(SIGTERM, handle_sigterm);
    const auto time_start = std::chrono::high_resolution_clock::now();

    std::string domain_path;
    std::string problem_path;
    std::string model_path;
    uint32_t batch_size;
    uint32_t min_expanded;
    double min_improvement;
    double batch_delta;
    bool use_wl;
    bool cpu;

    try
    {
        uint32_t DEFAULT_BATCH_SIZE = 256;
        uint32_t DEFAULT_MIN_EXPANDED = 1'000;
        double DEFAULT_MIN_IMPROVEMENT = 2.0;
        double DEFAULT_BATCH_DELTA = 1.0;

        // clang-format off
        TCLAP::CmdLine cmd("", ' ', "IPC 2023: Learning Track");
        TCLAP::ValueArg<std::string> domain_arg("", "domain", "Specifies the path to the domain file. If not provided, the file \"domain.pddl\" in the same directory as the given problem is assumed.", false, "", "path");
        TCLAP::ValueArg<std::string> problem_arg("", "problem", "Specifies the path to the problem file.", true, "", "path");
        TCLAP::ValueArg<std::string> model_arg("", "model", "Specifies the path to the model file.", true, "", "path");
        TCLAP::ValueArg<uint32_t> batch_size_arg("", "batch_size", "Sets the maximum number of successor states to be evaluated at once (default: " + std::to_string(DEFAULT_BATCH_SIZE) + ").", false, DEFAULT_BATCH_SIZE, "positive integer");
        TCLAP::ValueArg<uint32_t> min_expanded_arg("", "min_expanded", "Sets the minimum number of states to be expanded per iteration (default: " + std::to_string(DEFAULT_MIN_EXPANDED) + ").", false, DEFAULT_MIN_EXPANDED, "positive integer");
        TCLAP::ValueArg<double> min_improvement_arg("", "min_improvement", "Continue expanding states until a state improves the heuristic value by the specified amount (default: " + std::to_string(DEFAULT_MIN_IMPROVEMENT) + ").", false, DEFAULT_MIN_IMPROVEMENT, "positive integer");
        TCLAP::ValueArg<double> batch_delta_arg("", "batch_delta", "Add states to the batch whose f-value does not differ by more than the specified amount compared to the lowest f-value (default: " + std::to_string(DEFAULT_BATCH_DELTA) + ").", false, DEFAULT_BATCH_DELTA, "positive integer");
        TCLAP::SwitchArg use_wl_arg("", "wl", "Use WL to prune successor states.", false);
        TCLAP::SwitchArg cpu_arg("", "cpu", "Use CPU even if a GPU is available.", false);
        // clang-format on

        cmd.add(domain_arg);
        cmd.add(problem_arg);
        cmd.add(model_arg);
        cmd.add(batch_size_arg);
        cmd.add(min_expanded_arg);
        cmd.add(min_improvement_arg);
        cmd.add(batch_delta_arg);
        cmd.add(use_wl_arg);
        cmd.add(cpu_arg);

        cmd.parse(argc, argv);

        domain_path = domain_arg.getValue();
        problem_path = problem_arg.getValue();
        model_path = model_arg.getValue();
        batch_size = batch_size_arg.getValue();
        min_expanded = min_expanded_arg.getValue();
        min_improvement = min_improvement_arg.getValue();
        batch_delta = batch_delta_arg.getValue();
        use_wl = use_wl_arg.getValue();
        cpu = cpu_arg.getValue();
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

    auto successor_generator = planners::create_sucessor_generator(problem, planners::SuccessorGeneratorType::AUTOMATIC);
    planners::RelationalNeuralNetworkHeuristic heuristic(problem, model, 2 * batch_size);
    planners::GoalCondition is_goal(problem);

    auto search = [&](uint32_t min_expanded, std::chrono::high_resolution_clock::time_point& time_end, formalism::ActionList& output_plan) -> bool
    {
        formalism::State state = problem->initial;

        auto num_searches = 0;
        auto found_plan = false;

        while (!found_plan)
        {
            const auto search_start = std::chrono::high_resolution_clock::now();
            ++num_searches;

            planners::WeisfeilerLemanCondition weisfeiler(problem);
            planners::AlwaysCondition never(false);
            planners::ConditionBase& pruning = use_wl ? static_cast<planners::ConditionBase&>(weisfeiler) : static_cast<planners::ConditionBase&>(never);

            planners::SearchStateRepository state_repository;
            planners::SearchStatistics statistics;

            std::vector<formalism::Action> subplan;
            const planners::BatchedAstarSettings settings = { .batch_delta = batch_delta,
                                                              .min_improvement = min_improvement,
                                                              .batch_size = batch_size,
                                                              .min_expanded = min_expanded,
                                                              .max_expanded = std::numeric_limits<uint32_t>::max() };

            torch::NoGradGuard no_grad;
            found_plan = planners::batched_astar_search(settings,
                                                        state,
                                                        state_repository,
                                                        successor_generator,
                                                        heuristic,
                                                        is_goal,
                                                        pruning,
                                                        statistics,
                                                        time_end,
                                                        subplan);


            const auto search_end = std::chrono::high_resolution_clock::now();
            const auto search_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(search_end - search_start).count();

            if (std::chrono::high_resolution_clock::now() > time_end)
            {
                std::cout << "Ran out of time. Aborting." << std::endl;
                break;
            }

            if (!found_plan)
            {
                uint32_t best_state_index = 0;
                double best_heuristic_value = std::numeric_limits<double>::infinity();

                for (uint32_t state_index = 0; state_index < state_repository.num_indices(); ++state_index)
                {
                    const auto& context = state_repository.get_context(state_index);
                    const auto heuristic_value = context.heuristic_value;

                    if (heuristic_value < best_heuristic_value)
                    {
                        best_state_index = state_index;
                        best_heuristic_value = heuristic_value;
                    }
                }

                if (best_state_index == 0)
                {
                    std::cout << "Found no improvement from initial state" << std::endl;
                    break;
                }

                state = state_repository.get_state(best_state_index);
                subplan = state_repository.get_plan(best_state_index);

                const auto initial_value = state_repository.get_context(0).heuristic_value;
                const auto target_value = state_repository.get_context(best_state_index).heuristic_value;

                std::cout << std::fixed << std::setprecision(3);
                std::cout << "[" << num_searches << "] Subplan: " << subplan.size() << ", Heuristic: " << initial_value << " -> " << target_value;
                std::cout << ", Expanded: " << statistics.num_expanded << ", Generated: " << statistics.num_generated;
                std::cout << " (" << search_elapsed_ms << " ms)" << std::endl;
            }
            else
            {
                const auto initial_value = state_repository.get_context(0).heuristic_value;

                std::cout << std::fixed << std::setprecision(3);
                std::cout << "[" << num_searches << "] Subplan: " << subplan.size() << ", Heuristic: " << initial_value << " -> goal";
                std::cout << ", Expanded: " << statistics.num_expanded << ", Generated: " << statistics.num_generated;
                std::cout << " (" << search_elapsed_ms << " ms)" << std::endl;
            }

            output_plan.insert(output_plan.end(), subplan.begin(), subplan.end());
        }

        return found_plan;
    };

    int32_t plan_ending = 1;

    try
    {
        std::cout << "Attempting to find a plan by using the learned model as a policy..." << std::endl;
        const auto time_policy_start = std::chrono::high_resolution_clock::now();
        std::vector<formalism::Action> policy_plan;
        auto time_policy_deadline = time_start + std::chrono::minutes(16);
        const auto found_policy_plan = search(min_expanded, time_policy_deadline, policy_plan);
        const auto time_policy_end = std::chrono::high_resolution_clock::now();

        if (found_policy_plan)
        {
            const auto filename = "plan." + std::to_string(plan_ending);
            write_plan(filename, policy_plan);
            ++plan_ending;
        }

        const auto total_policy_seconds = std::chrono::duration_cast<std::chrono::seconds>(time_policy_end - time_policy_start).count();
        const auto policy_minutes = total_policy_seconds / 60;
        const auto policy_seconds = total_policy_seconds - policy_minutes * 60;
        std::cout << "Planned for " << policy_minutes << " minutes and " << policy_seconds << " seconds" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Search #1 crashed: " << std::endl;
        std::cerr << e.what() << std::endl;
        return 1;
    }

    try
    {
        std::cout << "Attempting to find plan by using the learned model as a heuristic..." << std::endl;
        const auto time_search_start = std::chrono::high_resolution_clock::now();
        std::vector<formalism::Action> search_plan;
        auto time_search_deadline = time_start + std::chrono::minutes(28);
        const auto found_search_plan = search(std::numeric_limits<uint32_t>::max(), time_search_deadline, search_plan);
        const auto time_search_end = std::chrono::high_resolution_clock::now();

        if (found_search_plan)
        {
            std::string filename = "plan." + std::to_string(plan_ending);
            write_plan(filename, search_plan);
            ++plan_ending;
        }

        const auto total_search_seconds = std::chrono::duration_cast<std::chrono::seconds>(time_search_end - time_search_start).count();
        const auto search_minutes = total_search_seconds / 60;
        const auto search_seconds = total_search_seconds - search_minutes * 60;
        std::cout << "Planned for " << search_minutes << " minutes and " << search_seconds << " seconds" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Search #2 crashed: " << std::endl;
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
