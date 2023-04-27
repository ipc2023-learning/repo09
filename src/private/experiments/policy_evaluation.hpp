#if !defined(EXPERIMENTS_POLICY_EVALUATION_HPP_)
#define EXPERIMENTS_POLICY_EVALUATION_HPP_

#include "../models/relational_neural_network.hpp"
#include "dataset_experiment.hpp"
#include "torch/torch.h"

#include <map>
#include <vector>

namespace experiments
{
    class PolicyEvaluation
    {
      private:
        uint32_t batch_size_;
        uint32_t max_epochs_;
        double learning_rate_;
        double discount_factor_;
        bool disable_balancing_;
        bool disable_baseline_;
        bool disable_value_regularization_;

      public:
        PolicyEvaluation(uint32_t batch_size,
                         uint32_t max_epochs,
                         double learning_rate,
                         double discount_factor,
                         bool disable_balancing,
                         bool disable_baseline,
                         bool disable_value_regularization) :
            batch_size_(batch_size),
            max_epochs_(max_epochs),
            learning_rate_(learning_rate),
            discount_factor_(discount_factor),
            disable_balancing_(disable_balancing),
            disable_baseline_(disable_baseline),
            disable_value_regularization_(disable_value_regularization) {};

        void fit(models::RelationalNeuralNetwork& model, const planners::StateSpaceList& training_set, const planners::StateSpaceList& validation_set);
    };
}  // namespace experiments

#endif  // EXPERIMENTS_POLICY_EVALUATION_HPP_
