#if !defined(EXPERIMENTS_POLICY_SUPERVISED_OPTIMAL_HPP_)
#define EXPERIMENTS_POLICY_SUPERVISED_OPTIMAL_HPP_

#include "../models/relational_neural_network.hpp"
#include "../planners/state_space.hpp"
#include "policy_dataset_experiment.hpp"

#include <map>
#include <vector>

namespace experiments
{
    class PolicySupervisedOptimal : public PolicyDatasetExperiment
    {
      private:
        double learning_rate_;
        bool disable_value_regularization_;

      protected:
        std::shared_ptr<torch::optim::Optimizer> create_optimizer(models::RelationalNeuralNetwork& model) override;

      public:
        PolicySupervisedOptimal(uint32_t batch_size, uint32_t max_epochs, double learning_rate, bool disable_balancing, bool disable_value_regularization) :
            PolicyDatasetExperiment(batch_size, batch_size, max_epochs, 1, 0.999, disable_balancing, true, true, PolicySamplingMethod::Policy),
            learning_rate_(learning_rate),
            disable_value_regularization_(disable_value_regularization) {};

        torch::Tensor train_loss(uint32_t epoch, const planners::StateSpaceSampleList& batch, const PolicyBatchOutput& output) override;
    };
}  // namespace experiments

#endif  // EXPERIMENTS_POLICY_SUPERVISED_OPTIMAL_HPP_
