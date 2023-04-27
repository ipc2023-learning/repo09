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
#include "private/experiments/fixed_horizon_reinforce.hpp"
#include "private/experiments/fixed_horizon_reinforce_baseline.hpp"
#include "private/experiments/rewards/blind_reward_function.hpp"
#include "private/experiments/rewards/optimal_reward_function.hpp"
#include "private/experiments/rewards/reward_function.hpp"
#include "private/formalism/problem.hpp"
#include "train_rl.hpp"

#include <iostream>

std::vector<std::string> reinforcement_learning_method_types() { return std::vector<std::string>({ "reinforce_baseline", "reinforce" }); }
std::vector<std::string> reinforcement_learning_reward_types() { return std::vector<std::string>({ "blind", "optimal" }); }

void train_rl(const ReinforcementLearningSettings& settings, const formalism::ProblemDescriptionList& problems, models::RelationalNeuralNetwork& model)
{
    planners::StateSpaceList small_state_spaces = compute_state_spaces(problems, 100'000, settings.use_weisfeiler_leman);

    std::sort(small_state_spaces.begin(),
              small_state_spaces.end(),
              [](const planners::StateSpace& lhs, const planners::StateSpace& rhs) -> bool { return lhs->num_states() < rhs->num_states(); });

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

    experiments::RewardFunction reward_function = nullptr;

    if (settings.reward == "blind")
    {
        reward_function = experiments::create_blind_reward_function();
    }
    else if (settings.reward == "optimal")
    {
        reward_function = experiments::create_optimal_reward_function();
    }
    else
    {
        std::cout << "[Internal Error] train_rl.cpp: This should have been caught by argument validation" << std::endl;
        return;
    }

    try
    {
        std::shared_ptr<experiments::MonteCarloExperiment> experiment(nullptr);

        if (settings.method == "reinforce_baseline")
        {
            experiment = std::make_shared<experiments::FixedHorizonReinforceBaseline>(settings.batch_size,
                                                                                      settings.max_epochs,
                                                                                      settings.learning_rate,
                                                                                      settings.horizon,
                                                                                      settings.discount_factor,
                                                                                      settings.bounds_factor,
                                                                                      reward_function);
        }
        else if (settings.method == "reinforce")
        {
            experiment = std::make_shared<experiments::FixedHorizonReinforce>(settings.batch_size,
                                                                              settings.max_epochs,
                                                                              settings.learning_rate,
                                                                              settings.horizon,
                                                                              settings.discount_factor,
                                                                              reward_function);
        }
        else
        {
            throw std::invalid_argument(settings.method + " is not implemented");
        }

        experiment->fit(model, training_state_spaces, validation_state_spaces);
    }
    catch (const std::exception& ex)
    {
        std::cout << "error: " << ex.what() << std::endl;
        return;
    }
    catch (...)
    {
        std::cout << "unknown error occurred" << std::endl;
        return;
    }
}
