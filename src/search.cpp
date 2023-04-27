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


#include "private/libraries/tclap/CmdLine.h"
#include "private/pddl/domain_parser.hpp"
#include "private/pddl/problem_parser.hpp"
#include "private/planners/bfs_search.hpp"

#include <boost/algorithm/string.hpp>

// Older versions of LibC++ does not have filesystem (e.g., ubuntu 18.04), use the experimental version
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

std::vector<std::string> successor_generator_types() { return std::vector<std::string>({ "automatic", "lifted", "grounded" }); }

int main(int argc, char* argv[])
{
    std::string domain_path;
    std::string problem_path;
    std::string successor_generator_type;

    const auto successor_generators = successor_generator_types();

    try
    {
        // clang-format off
        TCLAP::CmdLine cmd("", ' ', "0.1");
        TCLAP::ValueArg<std::string> domain_arg("", "domain", "Path to domain file", false, "", "path");
        TCLAP::ValueArg<std::string> problem_arg("", "problem", "Path to problem file", true, "", "path");
        TCLAP::ValueArg<std::string> successor_generator_arg("", "successor_generator", "One of [" + boost::algorithm::join(successor_generators, ", ") + "]", false, "automatic", "type");
        // clang-format off

        cmd.add(domain_arg);
        cmd.add(problem_arg);
        cmd.add(successor_generator_arg);

        cmd.parse(argc, argv);

        domain_path = domain_arg.getValue();
        problem_path = problem_arg.getValue();
        successor_generator_type = successor_generator_arg.getValue();

    }
    catch (TCLAP::ArgException& e)  // catch any exceptions
    {
        std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
        return 1;
    }

    if (!fs::exists(problem_path))
    {
        std::cout << "Error: \"problem\" does not exist" << std::endl;
        return 3;
    }

    if (domain_path.size() == 0)
    {
        domain_path = (fs::path(problem_path).parent_path() / "domain.pddl").string();
        std::cout << "Info: \"domain\" not given, trying \"" << domain_path << "\"" << std::endl;
    }

    if (!fs::exists(domain_path))
    {
        std::cout << "Error: \"domain\" does not exist" << std::endl;
        return 2;
    }

    if (std::count(successor_generators.begin(), successor_generators.end(), successor_generator_type) == 0)
    {
        std::cout << "Error: \"successor_generator\" does not exist" << std::endl;
        return 3;
    }

    // parse PDDL files

    parsers::DomainParser domain_parser(domain_path);
    const auto domain = domain_parser.parse();
    std::cout << domain << std::endl;

    parsers::ProblemParser problem_parser(problem_path);
    const auto problem = problem_parser.parse(domain);
    std::cout << problem << std::endl;

    // find  plan

    auto generator = planners::SuccessorGeneratorType::AUTOMATIC;

    if (successor_generator_type == "lifted")
    {
        generator = planners::SuccessorGeneratorType::LIFTED;
    }
    else if (successor_generator_type == "grounded")
    {
        generator = planners::SuccessorGeneratorType::GROUNDED;
    }

    planners::BreadthFirstSearch bfs(problem, generator);
    bfs.print_progress(true);
    std::vector<formalism::Action> plan;
    const auto found_solution = bfs.find_plan(plan);

    if (found_solution)
    {
        std::cout << "Found a solution!" << std::endl;

        for (const auto& action : plan)
        {
            std::cout << action << std::endl;
        }
    }
    else
    {
        std::cout << "Did not find a solution!" << std::endl;
    }

    std::cout << "Expanded: " << bfs.expanded << std::endl;
    std::cout << "Generated: " << bfs.generated << std::endl;
    std::cout << "Grounding Time: " << bfs.time_successors_ns / 1000000 << " ms" << std::endl;
    std::cout << "Total Time: " << bfs.time_total_ns / 1000000 << " ms" << std::endl;
}
