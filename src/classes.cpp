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


#include "private/algorithms/weisfeiler_leman.hpp"
#include "private/libraries/map_functions.hpp"
#include "private/libraries/tclap/CmdLine.h"
#include "private/pddl/domain_parser.hpp"
#include "private/pddl/problem_parser.hpp"
#include "private/planners/generators/successor_generator_factory.hpp"

#include <chrono>
#include <deque>
#include <iomanip>
#include <map>
#include <unordered_set>

// Older versions of LibC++ does not have filesystem (e.g., ubuntu 18.04), use the experimental version
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

namespace std
{
    template<>
    struct hash<pair<uint64_t, uint64_t>>
    {
        size_t operator()(const pair<uint64_t, uint64_t>& x) const { return (size_t) x.first; }
    };
}  // namespace std

formalism::AtomSet intersect(const formalism::AtomSet& lhs, const formalism::AtomSet& rhs)
{
    formalism::AtomSet intersection;

    for (const auto& atom : lhs)
    {
        if (rhs.find(atom) != rhs.end())
        {
            intersection.insert(atom);
        }
    }

    return intersection;
}

struct SearchFrame
{
    formalism::State state;
    uint32_t cost;

    SearchFrame(const formalism::State& state, uint32_t cost) : state(state), cost(cost) {}
};

int main(int argc, char* argv[])
{
    std::string domain_path;
    std::string problem_path;

    try
    {
        TCLAP::CmdLine cmd("", ' ', "0.1");
        TCLAP::ValueArg<std::string> domain_arg("", "domain", "Path to domain file", false, "", "path");
        TCLAP::ValueArg<std::string> problem_arg("", "problem", "Path to problem file", true, "", "path");

        cmd.add(domain_arg);
        cmd.add(problem_arg);
        cmd.parse(argc, argv);

        domain_path = domain_arg.getValue();
        problem_path = problem_arg.getValue();
    }
    catch (TCLAP::ArgException& e)  // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
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

    // parse PDDL files

    parsers::DomainParser domain_parser(domain_path);
    const auto domain = domain_parser.parse();
    std::cout << domain << std::endl;

    parsers::ProblemParser problem_parser(problem_path);
    const auto problem = problem_parser.parse(domain);
    std::cout << problem << std::endl;

    const auto successor_generator = planners::create_sucessor_generator(problem, planners::SuccessorGeneratorType::AUTOMATIC);
    std::deque<SearchFrame> queue({ SearchFrame(problem->initial, 0) });

    uint32_t generated = 0;
    uint32_t expanded = 0;
    uint32_t duplicates = 0;
    uint32_t last_cost = 0;
    uint32_t wl_duplicates = 0;

    const auto time_start = std::chrono::high_resolution_clock::now();

    std::unordered_set<formalism::State> closed;
    std::unordered_set<std::pair<uint64_t, uint64_t>> wl_closed;
    std::map<std::pair<uint64_t, uint64_t>, std::vector<formalism::State>> wl_classes;
    algorithms::WeisfeilerLeman wl(1);

    while (queue.size() > 0)
    {
        const auto frame = queue.front();
        queue.pop_front();

        if (closed.find(frame.state) != closed.end())
        {
            ++duplicates;
            continue;
        }
        else
        {
            closed.insert(frame.state);
        }

        const auto wl_graph = wl.to_wl_graph(problem, frame.state);
        const auto histogram = wl.compute_color_histogram(wl_graph);
        const auto hash = wl.hash_color_histogram(histogram);

        auto& class_list = get_or_add(wl_classes, hash);
        class_list.push_back(frame.state);

        if (wl_closed.find(hash) != wl_closed.end())
        {
            ++wl_duplicates;
        }
        else
        {
            wl_closed.insert(hash);
        }

        if (frame.cost > last_cost)
        {
            last_cost = frame.cost;
            const auto cost_time_end = std::chrono::high_resolution_clock::now();
            const auto cost_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cost_time_end - time_start).count();
            std::cout << "Cost: " << frame.cost << " (" << cost_time_ms << " ms)" << std::endl;
            std::cout << "Expanded: " << expanded << "; Generated: " << generated << "; Duplicates: " << duplicates << "; WL duplicates: " << wl_duplicates
                      << std::endl;
        }

        ++expanded;

        const auto ground_actions = successor_generator->get_applicable_actions(frame.state);

        for (const auto& ground_action : ground_actions)
        {
            ++generated;
            queue.push_back(SearchFrame(formalism::apply(ground_action, frame.state), frame.cost + 1));
        }
    }

    const auto time_end = std::chrono::high_resolution_clock::now();
    const auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    std::cout << "Time: " << time_ms << " ms" << std::endl << std::endl;

    std::cout << "[Static] ";
    for (const auto& atom : problem->initial->get_static_atoms())
    {
        std::cout << atom << " ";
    }
    std::cout << std::endl << std::endl;

    int32_t class_id = 1;
    for (const auto& kvp : wl_classes)
    {
        std::cout << "[Class] " << class_id << std::endl;

        const auto common_atoms_list = kvp.second[0]->get_dynamic_atoms();
        formalism::AtomSet common_atoms(common_atoms_list.begin(), common_atoms_list.end());

        for (const auto& state : kvp.second)
        {
            formalism::AtomSet next_common_atoms;
            const auto atoms_list = state->get_dynamic_atoms();
            formalism::AtomSet atoms(atoms_list.begin(), atoms_list.end());
            common_atoms = intersect(common_atoms, atoms);
        }

        std::cout << "[Common] ";
        for (const auto& atom : common_atoms)
        {
            std::cout << atom << " ";
        }
        std::cout << std::endl;

        for (const auto& state : kvp.second)
        {
            std::cout << "[State] ";

            for (const auto& atom : state->get_dynamic_atoms())
            {
                if (common_atoms.find(atom) == common_atoms.end())
                {
                    std::cout << atom << " ";
                }
            }

            std::cout << std::endl;
        }

        ++class_id;
        std::cout << std::endl;
    }
}
