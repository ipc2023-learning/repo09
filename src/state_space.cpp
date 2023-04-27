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


#include "private/planners/state_space.hpp"

#include "private/libraries/tclap/CmdLine.h"
#include "private/pddl/parsers.hpp"

#include <vector>

// Older versions of LibC++ does not have filesystem (e.g., ubuntu 18.04), use the experimental version
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

std::vector<formalism::ProblemDescription> load_problems(const fs::path& path)
{
    std::vector<formalism::ProblemDescription> problems;
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

        problems.push_back(problem);
    }

    return problems;
}

int main(int argc, char* argv[])
{
    std::string input_path;
    uint32_t max_expanded;
    bool simplify_wl;
    bool prune_wl;

    try
    {
        // clang-format off
        TCLAP::CmdLine cmd("", ' ', "0.1");
        TCLAP::ValueArg<std::string> input_arg("", "input", "Path to a problem", true, "", "path");
        TCLAP::ValueArg<uint32_t> max_expanded_arg("", "max_expanded", "Number of states to expand to evaluate difficulty of problem", false, 1000, "positive integer");
        TCLAP::SwitchArg simplify_wl_arg("", "simplify_wl", "use Weisfeiler-Leman to simplify the state space in post", false);
        TCLAP::SwitchArg prune_wl_arg("", "prune_wl", "use Weisfeiler-Leman to prune the state space", false);
        // clang-format on

        cmd.add(input_arg);
        cmd.add(max_expanded_arg);
        cmd.add(simplify_wl_arg);
        cmd.add(prune_wl_arg);
        cmd.parse(argc, argv);

        input_path = input_arg.getValue();
        max_expanded = max_expanded_arg.getValue();
        simplify_wl = simplify_wl_arg.getValue();
        prune_wl = prune_wl_arg.getValue();
    }
    catch (TCLAP::ArgException& e)  // catch any exceptions
    {
        std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
        return 1;
    }

    if (!fs::exists(input_path))
    {
        std::cerr << "Error: \"input\" does not exist" << std::endl;
        return 2;
    }

    if (fs::is_directory(input_path))
    {
        std::cerr << "Error: \"input\" is a directory" << std::endl;
        return 3;
    }

    const auto problem = load_problems(input_path).at(0);
    auto state_space = planners::create_state_space(problem, max_expanded, prune_wl);

    if (state_space && simplify_wl && !prune_wl)
    {
        state_space = planners::prune_state_space_with_weisfeiler_leman(state_space);
    }

    if (!state_space)
    {
        std::cout << "Could not process state space" << std::endl;
        return 1;
    }

    std::cout << "digraph " << problem->domain->name << std::endl;
    std::cout << "{" << std::endl;

    // Define nodes/states

    for (const auto& state : state_space->get_states())
    {
        const auto index = state_space->get_unique_index_of_state(state);
        std::cout << "    S" << index << " [shape=";

        const auto is_goal = state_space->is_goal_state(state);
        const auto is_dead_end = state_space->is_dead_end_state(state);

        if (is_goal)
        {
            std::cout << "circle";
        }
        else if (is_dead_end)
        {
            std::cout << "diamond";
        }
        else
        {
            std::cout << "box";
        }

        const auto f = state_space->get_distance_from_initial_state(state);
        const auto h = is_dead_end ? -1 : state_space->get_distance_to_goal_state(state);

        std::cout << ", label = \""
                  << "f = " << f << ", h = " << h << "\"]" << std::endl;
    }

    // Define edges/transitions

    for (const auto& state : state_space->get_states())
    {
        for (const auto& transition : state_space->get_forward_transitions(state))
        {
            const auto source_index = state_space->get_unique_index_of_state(transition->source_state);
            const auto target_index = state_space->get_unique_index_of_state(transition->target_state);
            std::cout << "    S" << source_index << " -> S" << target_index << " [label=\"" << transition->action << "\"]" << std::endl;
        }
    }

    std::cout << "}" << std::endl;
}
