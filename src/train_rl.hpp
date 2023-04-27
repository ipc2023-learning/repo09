#if !defined(TRAIN_REINFORCEMENT_LEARNING_HPP_)
#define TRAIN_REINFORCEMENT_LEARNING_HPP_

#include "private/formalism/declarations.hpp"
#include "private/models/relational_neural_network.hpp"

#include <vector>

std::vector<std::string> reinforcement_learning_method_types();
std::vector<std::string> reinforcement_learning_reward_types();

struct ReinforcementLearningSettings
{
    std::string method;
    std::string reward;
    uint32_t batch_size;
    uint32_t max_epochs;
    uint32_t horizon;
    double learning_rate;
    double discount_factor;
    double bounds_factor;
    bool use_weisfeiler_leman;
};

void train_rl(const ReinforcementLearningSettings& settings, const formalism::ProblemDescriptionList& problems, models::RelationalNeuralNetwork& model);

#endif  // TRAIN_REINFORCEMENT_LEARNING_HPP_
