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
#include "private/datasets/balanced_dataset.hpp"
#include "private/datasets/random_dataset.hpp"
#include "private/experiments/policy_actor_critic.hpp"
#include "private/experiments/policy_evaluation.hpp"
#include "private/experiments/policy_iterative_tabular_evaluation.hpp"
#include "private/experiments/policy_iterative_tabular_probability.hpp"
#include "private/experiments/policy_model_actor_critic.hpp"
#include "private/experiments/policy_probability.hpp"
#include "private/experiments/policy_suboptimal_model_actor_critic.hpp"
#include "private/experiments/policy_supervised_optimal.hpp"
#include "private/experiments/policy_tabular_model_actor_critic.hpp"
#include "private/formalism/declarations.hpp"
#include "private/formalism/problem.hpp"
#include "private/planners/state_space.hpp"
#include "train_ol.hpp"

#include <iostream>
std::vector<std::string> other_method_types()

{
    return std::vector<std::string>({
        "policy_supervised_optimal",
        "policy_evaluation",
        "policy_actor_critic",
        "policy_actor_critic_full",
        "policy_model_actor_critic",
        "policy_model_actor_critic_uniform",
        "policy_suboptimal_model_actor_critic",
        "policy_tabular_model_actor_critic",
        "policy_iterative_tabular_evaluation",
        "policy_probability",
        "policy_iterative_tabular_probability",
    });
}

void train_ol(const OtherLearningSettings& settings, const formalism::ProblemDescriptionList& problems, models::RelationalNeuralNetwork& model)
{
    planners::StateSpaceList small_state_spaces = compute_state_spaces(problems, 100'000, settings.use_weisfeiler_leman);

    std::sort(small_state_spaces.begin(), small_state_spaces.end(), [](const planners::StateSpace& lhs, const planners::StateSpace& rhs) -> bool {
        return lhs->num_states() < rhs->num_states();
    });

    const auto training_size = (small_state_spaces.size() == 1) ? 1 : std::min(small_state_spaces.size() - 1, (std::size_t)(0.9 * small_state_spaces.size()));

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

    try
    {
        if (settings.method == "policy_evaluation")
        {
            experiments::PolicyEvaluation experiment(settings.batch_size,
                                                     settings.max_epochs,
                                                     settings.learning_rate,
                                                     settings.discount_factor,
                                                     settings.disable_balancing,
                                                     settings.disable_baseline,
                                                     settings.disable_value_regularization);
            experiment.fit(model, training_state_spaces, validation_state_spaces);
        }
        else if (settings.method == "policy_probability")
        {
            experiments::PolicyProbability experiment(settings.batch_size,
                                                      settings.max_epochs,
                                                      settings.learning_rate,
                                                      settings.discount_factor,
                                                      settings.disable_balancing,
                                                      settings.disable_baseline,
                                                      settings.disable_value_regularization);
            experiment.fit(model, training_state_spaces, validation_state_spaces);
        }
        else if (settings.method == "policy_iterative_tabular_evaluation")
        {
            experiments::PolicyIterativeTabularEvaluation experiment(settings.batch_size,
                                                                     settings.chunk_size,
                                                                     settings.max_epochs,
                                                                     settings.trajectory_length,
                                                                     settings.learning_rate,
                                                                     settings.discount_factor,
                                                                     settings.disable_balancing,
                                                                     settings.disable_baseline,
                                                                     settings.disable_value_regularization);

            experiment.fit(model, training_state_spaces, validation_state_spaces);
        }
        else if (settings.method == "policy_iterative_tabular_probability")
        {
            experiments::PolicyIterativeTabularProbability experiment(settings.batch_size,
                                                                      settings.chunk_size,
                                                                      settings.max_epochs,
                                                                      settings.learning_rate,
                                                                      settings.discount_factor,
                                                                      settings.disable_balancing,
                                                                      settings.disable_baseline,
                                                                      settings.disable_value_regularization);
            experiment.fit(model, training_state_spaces, validation_state_spaces);
        }
        else if (settings.method == "policy_actor_critic")
        {
            experiments::PolicyActorCritic experiment(settings.batch_size,
                                                      settings.chunk_size,
                                                      settings.max_epochs,
                                                      settings.trajectory_length,
                                                      settings.learning_rate,
                                                      settings.discount_factor,
                                                      settings.disable_balancing,
                                                      settings.disable_baseline,
                                                      settings.disable_value_regularization,
                                                      false);

            experiment.fit(model, training_state_spaces, validation_state_spaces);
        }
        else if (settings.method == "policy_actor_critic_full")
        {
            experiments::PolicyActorCritic experiment(settings.batch_size,
                                                      settings.chunk_size,
                                                      settings.max_epochs,
                                                      settings.trajectory_length,
                                                      settings.learning_rate,
                                                      settings.discount_factor,
                                                      settings.disable_balancing,
                                                      settings.disable_baseline,
                                                      settings.disable_value_regularization,
                                                      true);

            experiment.fit(model, training_state_spaces, validation_state_spaces);
        }
        else if (settings.method == "policy_model_actor_critic")
        {
            experiments::PolicyModelActorCritic experiment(settings.batch_size,
                                                           settings.chunk_size,
                                                           settings.max_epochs,
                                                           settings.trajectory_length,
                                                           settings.learning_rate,
                                                           settings.discount_factor,
                                                           settings.disable_balancing,
                                                           settings.disable_baseline,
                                                           settings.disable_value_regularization,
                                                           false,
                                                           experiments::PolicySamplingMethod::Uniform);

            experiment.fit(model, training_state_spaces, validation_state_spaces);
        }
        else if (settings.method == "policy_model_actor_critic_uniform")
        {
            experiments::PolicyModelActorCritic experiment(settings.batch_size,
                                                           settings.chunk_size,
                                                           settings.max_epochs,
                                                           settings.trajectory_length,
                                                           settings.learning_rate,
                                                           settings.discount_factor,
                                                           settings.disable_balancing,
                                                           settings.disable_baseline,
                                                           settings.disable_value_regularization,
                                                           true,
                                                           experiments::PolicySamplingMethod::Uniform);

            experiment.fit(model, training_state_spaces, validation_state_spaces);
        }
        else if (settings.method == "policy_tabular_model_actor_critic")
        {
            experiments::PolicyTabularModelActorCritic experiment(settings.batch_size,
                                                                  settings.chunk_size,
                                                                  settings.max_epochs,
                                                                  settings.trajectory_length,
                                                                  settings.learning_rate,
                                                                  settings.discount_factor,
                                                                  settings.disable_balancing,
                                                                  settings.disable_baseline);

            experiment.fit(model, training_state_spaces, validation_state_spaces);
        }
        else if (settings.method == "policy_suboptimal_model_actor_critic")
        {
            experiments::PolicyIterativeFullProbability experiment(settings.batch_size,
                                                                   settings.chunk_size,
                                                                   settings.max_epochs,
                                                                   settings.trajectory_length,
                                                                   settings.learning_rate,
                                                                   settings.discount_factor,
                                                                   settings.disable_balancing,
                                                                   settings.disable_baseline,
                                                                   settings.disable_value_regularization);

            experiment.fit(model, training_state_spaces, validation_state_spaces);
        }
        else if (settings.method == "policy_supervised_optimal")
        {
            experiments::PolicySupervisedOptimal experiment(settings.batch_size,
                                                            settings.max_epochs,
                                                            settings.learning_rate,
                                                            settings.disable_balancing,
                                                            settings.disable_value_regularization);
            experiment.fit(model, training_state_spaces, validation_state_spaces);
        }
        else
        {
            throw std::invalid_argument(settings.method + " is not implemented");
        }
    }
    catch (const std::exception& ex)
    {
        std::cout << "error: " << ex.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "unknown error occurred" << std::endl;
    }
}
