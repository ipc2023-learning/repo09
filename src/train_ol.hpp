#if !defined(TRAIN_OTHER_LEARNING_HPP_)
#define TRAIN_OTHER_LEARNING_HPP_

#include "private/formalism/declarations.hpp"
#include "private/models/relational_neural_network.hpp"

#include <vector>

std::vector<std::string> other_method_types();

struct OtherLearningSettings
{
    std::string method;
    uint32_t batch_size;
    uint32_t chunk_size;
    uint32_t max_epochs;
    uint32_t trajectory_length;
    int32_t horizon;
    double learning_rate;
    double discount_factor;
    bool disable_balancing;
    bool disable_baseline;
    bool disable_value_regularization;
    bool use_weisfeiler_leman;
};

void train_ol(const OtherLearningSettings& settings, const formalism::ProblemDescriptionList& problems, models::RelationalNeuralNetwork& model);

#endif  // TRAIN_OTHER_LEARNING_HPP_
