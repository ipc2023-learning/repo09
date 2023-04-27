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
#include "private/planners/generators/lifted_successor_generator.hpp"
#include "private/planners/state_space.hpp"

#include <set>

// Older versions of LibC++ does not have filesystem (e.g., ubuntu 18.04), use the experimental version
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

int32_t greedy_depot(const planners::StateSpace& state_space)
{
    const auto initial_state = state_space->get_initial_state();
    const auto predicate_map = state_space->domain->get_predicate_map();
    const auto atoms_by_predicate = initial_state->get_atoms_grouped_by_predicate();

    const auto crate_list = atoms_by_predicate.at(predicate_map.at("crate"));
    const auto place_list = atoms_by_predicate.at(predicate_map.at("place"));

    int32_t length = 0;

    // Drive to each location and load up all crates (on a single truck)

    length += place_list.size() - 1;
    length += 2 * crate_list.size();

    // Drive to each palce and unload all packages

    length += place_list.size() - 1;
    length += 2 * crate_list.size();

    return length;
}

int32_t greedy_satellite(const planners::StateSpace& state_space)
{
    const auto initial_state = state_space->get_initial_state();
    const auto predicate_map = state_space->domain->get_predicate_map();
    const auto atoms_by_predicate = initial_state->get_atoms_grouped_by_predicate();
    const auto goal_state = formalism::create_state(formalism::as_atoms(state_space->problem->goal), state_space->problem);
    const auto goal_atoms_by_predicate = goal_state->get_atoms_grouped_by_predicate();

    const auto modes = atoms_by_predicate.at(predicate_map.at("mode"));

    int32_t length = 0;

    const auto predicate_have_image = predicate_map.at("have_image");
    if (goal_atoms_by_predicate.find(predicate_have_image) != goal_atoms_by_predicate.end())
    {
        const auto goal_have_image_list = goal_atoms_by_predicate.at(predicate_have_image);
        length += 6 * goal_have_image_list.size();

        for (const auto& mode : modes)
        {
            const auto shared_directions = formalism::filter(goal_have_image_list, mode->arguments.at(0), 1);

            if (shared_directions.size() > 1)
            {
                length -= 4 * (shared_directions.size() - 1);
            }
        }
    }

    const auto predicate_pointing = predicate_map.at("pointing");
    if (goal_atoms_by_predicate.find(predicate_pointing) != goal_atoms_by_predicate.end())
    {
        const auto goal_pointing_list = goal_atoms_by_predicate.at(predicate_pointing);
        length += goal_pointing_list.size();
    }

    return length;
}

int32_t greedy_miconic(const planners::StateSpace& state_space)
{
    const auto initial_state = state_space->get_initial_state();
    const auto predicate_map = state_space->domain->get_predicate_map();
    const auto type_map = state_space->domain->get_type_map();
    const auto atoms_by_predicate = initial_state->get_atoms_grouped_by_predicate();

    const auto origin_floor_list = formalism::get_objects(atoms_by_predicate.at(predicate_map.at("origin")), 1);
    const auto destin_floor_list = formalism::get_objects(atoms_by_predicate.at(predicate_map.at("destin")), 1);
    const auto passenger_list = formalism::filter(state_space->problem->objects, type_map.at("passenger"));

    const auto origin_floor_set = std::set<formalism::Object>(origin_floor_list.begin(), origin_floor_list.end());
    const auto destin_floor_set = std::set<formalism::Object>(destin_floor_list.begin(), destin_floor_list.end());

    // Go to each floor that has at least one passenger and board the passenger(s); then go to each destination floor and depart passenger(s).

    return origin_floor_set.size() + 2 * passenger_list.size() + destin_floor_set.size();
}

int32_t greedy_logistics(const planners::StateSpace& state_space)
{
    const auto predicate_map = state_space->domain->get_predicate_map();
    const auto goal_state = formalism::create_state(formalism::as_atoms(state_space->problem->goal), state_space->problem);
    const auto goal_atoms_by_predicate = goal_state->get_atoms_grouped_by_predicate();
    const auto state_atoms_by_predicate = state_space->get_initial_state()->get_atoms_grouped_by_predicate();

    const auto goal_all_at_atoms = goal_atoms_by_predicate.at(predicate_map.at("at"));
    const auto at_atoms = state_atoms_by_predicate.at(predicate_map.at("at"));
    const auto goal_at_atoms = formalism::exclude(goal_all_at_atoms, at_atoms);
    const auto in_city_atoms = state_atoms_by_predicate.at(predicate_map.at("in-city"));

    const auto packages = formalism::get_objects(goal_at_atoms, 0);
    const auto destination_locations = formalism::get_unique_objects(goal_at_atoms, 1);
    const auto origin_locations = formalism::get_unique_objects(formalism::filter(at_atoms, packages, 0), 1);
    const auto origin_cities = formalism::get_unique_objects(formalism::filter(in_city_atoms, origin_locations, 0), 1);
    const auto destination_cities = formalism::get_unique_objects(formalism::filter(in_city_atoms, destination_locations, 0), 1);

    // TODO: Consider two cases: packages that needs to be delivered to different cities and packages that need to be delivered within the same city

    int32_t length = 0;
    length += 2 * origin_locations.size();                  // Drive to package location and then to airport
    length += 2 * packages.size();                          // Pick up each package and drop it off at the airport
    length += origin_cities.size() + packages.size();       // Fly to each origin city and pick up packages
    length += destination_cities.size() + packages.size();  // Fly to each destination city and drop off packages
    length += 2 * destination_locations.size();             // Drive to airport and then to destination location
    length += 2 * packages.size();                          // Pick up each package at the airport and then drop it off at destination
    return length;
}

// int32_t greedy_floortile(const planners::StateSpace& state_space) {}

int main(int argc, char* argv[])
{
    std::string domain_path;
    std::string problem_path;
    bool verbose;

    try
    {
        TCLAP::CmdLine cmd("", ' ', "0.1");
        TCLAP::ValueArg<std::string> domain_arg("", "domain", "Path to domain file", false, "", "path");
        TCLAP::ValueArg<std::string> problem_arg("", "problem", "Path to problem file", true, "", "path");
        TCLAP::SwitchArg verbose_arg("", "verbose", "Verbose printing", false);

        cmd.add(domain_arg);
        cmd.add(problem_arg);
        cmd.add(verbose_arg);

        cmd.parse(argc, argv);

        domain_path = domain_arg.getValue();
        problem_path = problem_arg.getValue();
        verbose = verbose_arg.getValue();
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
    }

    if (!fs::exists(domain_path))
    {
        std::cout << "Error: \"domain\" does not exist" << std::endl;
        return 2;
    }

    // parse PDDL files

    parsers::DomainParser domain_parser(domain_path);
    const auto domain = domain_parser.parse();

    parsers::ProblemParser problem_parser(problem_path);
    const auto problem = problem_parser.parse(domain);

    // expand state-space

    const auto state_space = planners::create_state_space(problem, 100'000, true);

    if (state_space)
    {
        int32_t greedy_length;
        const auto& domain_name = problem->domain->name;
        if (domain_name.find("depot") != std::string::npos)
        {
            greedy_length = greedy_depot(state_space);
        }
        else if (domain_name.find("satellite") != std::string::npos)
        {
            greedy_length = greedy_satellite(state_space);
        }
        else if (domain_name.find("miconic") != std::string::npos)
        {
            greedy_length = greedy_miconic(state_space);
        }
        else if (domain_name.find("logistics") != std::string::npos)
        {
            greedy_length = greedy_logistics(state_space);
        }
        else
        {
            std::cout << "Unknown approximation for: " << domain_name << std::endl;
            return 3;
        }

        const auto optimal_length = state_space->get_distance_to_goal_state(problem->initial);
        const auto ratio = (greedy_length == optimal_length) ? 1.0 : ((double) greedy_length / (double) optimal_length);

        if (verbose)
        {
            std::cout << "States: " << state_space->num_states() << std::endl;
            std::cout << "Transitions: " << state_space->num_transitions() << std::endl;
            std::cout << "Optimal: " << optimal_length << std::endl;
            std::cout << "Greedy: " << greedy_length << std::endl;
            std::cout << "Difference: " << greedy_length - optimal_length << std::endl;
            std::cout << "Ratio: " << ratio << std::endl;
        }

        std::cout << ratio << std::endl;

        if (ratio > 1.5)
        {
            return 0;
        }
        else
        {
            return 5;
        }
    }
    else
    {
        std::cout << "Number of reachable states exceeds 100 000." << std::endl;
        return 4;
    }
}
