#if !defined(TRAIN_DATASET_LEARNING_HPP_)
#define TRAIN_DATASET_LEARNING_HPP_

#include "private/formalism/declarations.hpp"
#include "private/models/relational_neural_network.hpp"

#include <vector>

std::vector<std::string> dataset_method_types();

struct DatasetLearningSettings
{
    std::string method;
    uint32_t batch_size;
    uint32_t chunk_size;
    uint32_t max_epochs;
    double learning_rate;
    double bounds_factor;
    bool disable_balancing;
    bool use_weisfeiler_leman;
};

void train_dl(const DatasetLearningSettings& settings, const formalism::ProblemDescriptionList& problems, models::RelationalNeuralNetwork& model);

#endif  // TRAIN_DATASET_LEARNING_HPP_
