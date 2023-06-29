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
#include "private/formalism/problem.hpp"
#include "private/models/relational_neural_network.hpp"
#include "private/models/utils.hpp"
#include "private/pddl/domain_parser.hpp"
#include "private/pddl/problem_parser.hpp"
#include "private/planners/heuristics/h2.hpp"

#include <random>

bool contains(const std::vector<std::string>& strings, const std::string& string)
{
    for (const auto& other_string : strings)
    {
        if (string == other_string)
        {
            return true;
        }
    }

    return false;
}

torch::Device load_device(bool force_cpu)
{
    if (torch::cuda::is_available() && !force_cpu)
    {
        std::cout << "GPU/CUDA device is available and will be used" << std::endl;
        return torch::kCUDA;
    }
    else if (torch::cuda::is_available() && !force_cpu)
    {
        std::cout << "GPU/CUDA device is available but will NOT be used" << std::endl;
        return torch::kCPU;
    }
    else
    {
        std::cout << "Only CPU is available and will be used" << std::endl;
        return torch::kCPU;
    }
}

models::RelationalNeuralNetwork load_model(const std::string& path,
                                           const std::string& type,
                                           const formalism::PredicateList& predicates,
                                           const models::DerivedPredicateList& derived_predicates,
                                           const formalism::TypeList& types,
                                           int32_t num_features,
                                           int32_t num_layers,
                                           bool global_readout,
                                           double maximum_smoothness)
{
    std::vector<std::pair<std::string, int32_t>> predicate_name_arities;

    for (const auto& predicate : predicates)
    {
        predicate_name_arities.push_back(std::make_pair(predicate->name, predicate->arity));
    }

    for (const auto& type : types)
    {
        if (type->name != "object")
        {
            predicate_name_arities.push_back(std::make_pair(type->name, 1));
        }
    }

    return load_model(path, type, predicate_name_arities, derived_predicates, num_features, num_layers, global_readout, maximum_smoothness);
}

models::RelationalNeuralNetwork load_model(const std::string& path,
                                           const std::string& type,
                                           const std::vector<std::pair<std::string, int32_t>>& predicates,
                                           const models::DerivedPredicateList& derived_predicates,
                                           int32_t num_features,
                                           int32_t num_layers,
                                           bool global_readout,
                                           double maximum_smoothness)
{
    models::RelationalNeuralNetwork model;
    if (models::load_model(&model, path))
    {
        std::cout << "Loaded existing model:" << std::endl;
        std::cout << model;
    }
    else
    {
        if (type == "relational_mpnn")
        {
            model = models::RelationalNeuralNetwork(
                models::RelationalMessagePassingNeuralNetwork(predicates, derived_predicates, num_features, num_layers, global_readout, maximum_smoothness));
        }
        else if (type == "relational_transformer")
        {
            model = models::RelationalNeuralNetwork(models::RelationalTransformer(predicates, derived_predicates, 10, 100, 512, 32));
        }

        std::cout << "Created new model:" << std::endl;
        std::cout << model;
    }

    uint64_t num_parameters = 0;
    for (auto parameter : model.parameters())
    {
        num_parameters += parameter.numel();
    }
    std::cout << "Number of parameters: " << num_parameters << std::endl << std::endl;

    return model;
}

formalism::ProblemDescriptionList load_problems(const fs::path& path)
{
    std::cout << "Parsing problems... " << std::endl;

    formalism::ProblemDescriptionList problems;
    std::vector<std::pair<fs::path, fs::path>> files;

    if (fs::is_directory(path))
    {
        for (const auto& file : fs::directory_iterator(path))
        {
            const auto file_path = file.path();

            if ((file_path.string().find("domain") == std::string::npos) && file_path.has_extension() && !file_path.extension().string().compare(".pddl"))
            {
                const auto problem_file = file_path;
                const auto domain_file = problem_file.parent_path() / "domain.pddl";

                if (fs::exists(problem_file) && fs::exists(domain_file))
                {
                    files.push_back(std::make_pair(domain_file, problem_file));
                }
            }
        }
    }
    else if ((path.string().find("domain") == std::string::npos) && path.has_extension() && !path.extension().string().compare(".pddl"))
    {
        const auto problem_file = path;
        const auto domain_file = problem_file.parent_path() / "domain.pddl";

        if (fs::exists(problem_file) && fs::exists(domain_file))
        {
            files.push_back(std::make_pair(domain_file, problem_file));
        }
    }

    for (const auto& file : files)
    {
        const auto& domain_path = file.first;
        const auto& problem_path = file.second;

        parsers::DomainParser domain_parser(domain_path);
        parsers::ProblemParser problem_parser(problem_path);

        const auto domain = domain_parser.parse();
        const auto problem = problem_parser.parse(domain);

        if (formalism::atoms_hold(formalism::as_atoms(problem->goal), problem->initial))
        {
            std::cout << "Problem \"" << problem->name << "\": The initial state is a goal state, skipping" << std::endl;
        }
        else
        {
            problems.push_back(problem);
        }
    }
    std::cout << std::endl;

    std::sort(problems.begin(),
              problems.end(),
              [](const formalism::ProblemDescription& lhs, const formalism::ProblemDescription& rhs) { return lhs->num_objects() < rhs->num_objects(); });

    std::cout << "Parsed " << problems.size() << " problems" << std::endl;

    return problems;
}

planners::StateSpaceList compute_state_spaces(const formalism::ProblemDescriptionList& problems,
                                              uint32_t max_size,
                                              bool use_weisfeiler_leman,
                                              bool& pruning_is_safe,
                                              bool& pruning_is_useful,
                                              int32_t timeout_s,
                                              int32_t max_memory_mb)
{
    // Generate training data.
    // We assume that if problem P_i is too large to expand, then all P_j, i < j, are also too large to expand.
    // This is to avoid high memory peak usages.

    planners::StateSpaceList small_state_spaces;
    formalism::ProblemDescriptionList large_problems;
    pruning_is_safe = true;
    pruning_is_useful = false;

    for (std::size_t problem_index = 0; problem_index < problems.size(); ++problem_index)
    {
        const auto& problem = problems[problem_index];
        planners::StateSpace state_space = nullptr;

        {
            const auto time_start = std::chrono::high_resolution_clock::now();
            state_space = planners::create_state_space(problem, max_size, false, timeout_s, max_memory_mb);
            const auto time_stop = std::chrono::high_resolution_clock::now();
            const auto time_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_stop - time_start).count();
            const auto num_objects = problem->objects.size();

            if (state_space)
            {
                const auto unsolvable = state_space->is_dead_end_state(problem->initial);
                const auto distance_from_initial = unsolvable ? -1 : state_space->get_distance_to_goal_state(problem->initial);

                std::cout << "Problem \"" << problem->name << "\" was fully expanded (" << state_space->num_states() << " states, "
                          << state_space->num_transitions() << " transitions, " << num_objects << " objects, " << time_elapsed_ms << " ms, "
                          << (unsolvable ? "unsolvable" : (std::to_string(distance_from_initial) + " steps to solve")) << ")";

                if ((distance_from_initial == 0) || unsolvable)
                {
                    state_space = nullptr;
                }
            }
            else
            {
                large_problems.push_back(problem);
                std::cout << "Problem \"" << problem->name << "\" has too many states, " << num_objects << " objects (" << time_elapsed_ms << " ms)";
                std::cout << "Adding remaining problems as large problems..." << std::endl;

                for (std::size_t index = problem_index + 1; index < problems.size(); ++index)
                {
                    large_problems.push_back(problems[index]);
                }

                break;
            }
        }

        if (state_space && use_weisfeiler_leman && pruning_is_safe)
        {
            const auto pruned_state_space = planners::prune_state_space_with_weisfeiler_leman(state_space);

            if (pruned_state_space)
            {
                std::cout << ", from " << std::to_string(state_space->num_states()) << " (" << state_space->num_transitions() << ")"
                          << " to " << std::to_string(pruned_state_space->num_states()) << " (" << pruned_state_space->num_transitions() << ")"
                          << " states (transitions)";

                if (state_space->num_states() != pruned_state_space->num_states())
                {
                    pruning_is_useful = true;
                }

                state_space = pruned_state_space;
            }
            else
            {
                std::cout << ", pruning is not safe";
                pruning_is_safe = false;
            }
        }

        if (state_space)
        {
            small_state_spaces.push_back(state_space);
        }

        std::cout << std::endl;
    }

    if (use_weisfeiler_leman)
    {
        std::cout << std::endl;
        std::cout << (pruning_is_safe ? "1-WL Pruning appears safe" : "1-WL Pruning is not safe") << std::endl;
        std::cout << (pruning_is_useful ? "1-WL Pruning is useful" : "1-WL Pruning appears not be useful") << std::endl;
        std::cout << std::endl;
    }

    if (use_weisfeiler_leman && pruning_is_safe && pruning_is_useful)
    {
        for (const auto& problem : large_problems)
        {
            const auto time_start = std::chrono::high_resolution_clock::now();
            const auto state_space = planners::create_state_space(problem, max_size, true, timeout_s, max_memory_mb);
            const auto time_stop = std::chrono::high_resolution_clock::now();
            const auto time_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_stop - time_start).count();

            if (state_space)
            {
                small_state_spaces.push_back(state_space);
                std::cout << "Problem \"" << problem->name << "\" was fully expanded with 1-WL (" << time_elapsed_ms << " ms), "
                          << std::to_string(state_space->num_states()) << " (" << std::to_string(state_space->num_transitions()) << ") states (transitions)"
                          << std::endl;
            }
            else
            {
                std::cout << "Problem \"" << problem->name << "\" has too many states with 1-WL (" << time_elapsed_ms << " ms)" << std::endl;
                std::cout << "Skip remaining large problems..." << std::endl;
                break;
            }
        }
    }

    return small_state_spaces;
}

planners::StateSpaceList
compute_state_spaces(const formalism::ProblemDescriptionList& problems, uint32_t max_size, bool use_weisfeiler_leman, int32_t timeout_s, int32_t max_memory_mb)
{
    bool pruning_is_safe;
    bool pruning_is_useful;
    return compute_state_spaces(problems, max_size, use_weisfeiler_leman, pruning_is_safe, pruning_is_useful, timeout_s, max_memory_mb);
}
