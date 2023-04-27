#if !defined(EXPERIMENTS_POLICY_EVALUATION_UTILS_HPP_)
#define EXPERIMENTS_POLICY_EVALUATION_UTILS_HPP_

#include "../models/relational_neural_network.hpp"
#include "../planners/state_space.hpp"

#include <map>
#include <vector>

namespace experiments
{
    struct TransitionProbabilities
    {
        std::vector<std::vector<int64_t>> neighbor_indices;
        std::vector<std::vector<double>> probabilities;
        std::vector<formalism::State> states;
        std::vector<bool> is_goal_state;
        std::vector<bool> is_dead_end_state;
    };

    TransitionProbabilities create_transition_probabilities(const planners::StateSpace& state_space);

    std::pair<std::vector<formalism::StateTransitionsVector>, std::vector<std::vector<std::size_t>>>
    create_batches(const formalism::ProblemDescription& problem, const TransitionProbabilities& transition_probabilities, const uint32_t batch_size);

    void update_transition_probabilities(const std::vector<formalism::StateTransitionsVector>& batches,
                                         const std::vector<std::vector<std::size_t>>& state_index_list,
                                         TransitionProbabilities& transition_probabilities,
                                         models::RelationalNeuralNetwork& model);

    std::vector<double> compute_optimal_policy_evaluation(const planners::StateSpace& state_space, const double discount_factor);

    std::vector<double> compute_policy_evaluation(const TransitionProbabilities& transition_probabilities, const double discount_factor);

    std::vector<double> compute_probability_evaluation(const TransitionProbabilities& transition_probabilities, const std::vector<double>& values);
}  // namespace experiments

#endif  // EXPERIMENTS_POLICY_EVALUATION_UTILS_HPP_
