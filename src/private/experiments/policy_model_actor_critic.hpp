#if !defined(EXPERIMENTS_POLICY_MODEL_ACTOR_CRITIC_HPP_)
#define EXPERIMENTS_POLICY_MODEL_ACTOR_CRITIC_HPP_

#include "../models/relational_neural_network.hpp"
#include "policy_dataset_experiment.hpp"
#include "torch/torch.h"

#include <map>
#include <vector>

namespace experiments
{
    class PolicyModelActorCritic : public PolicyDatasetExperiment
    {
      private:
        double learning_rate_;
        bool disable_baseline_;
        bool disable_value_regularization_;
        bool sample_successor_;

        torch::Tensor loss(uint32_t epoch, const planners::StateSpaceSampleList& batch, const PolicyBatchOutput& output);

      protected:
        std::shared_ptr<torch::optim::Optimizer> create_optimizer(models::RelationalNeuralNetwork& model) override;

        torch::Tensor train_loss(uint32_t epoch, const planners::StateSpaceSampleList& batch, const PolicyBatchOutput& output) override;

        planners::StateSpaceSampleList
        get_batch(const std::shared_ptr<datasets::Dataset>& set, uint32_t epoch, uint32_t batch_index, uint32_t batch_size) override;

      public:
        PolicyModelActorCritic(uint32_t batch_size,
                               uint32_t chunk_size,
                               uint32_t max_epochs,
                               uint32_t trajectory_length,
                               double learning_rate,
                               double discount_factor,
                               bool disable_balancing,
                               bool disable_baseline,
                               bool disable_value_regularization,
                               bool sample_successor,
                               PolicySamplingMethod sample_method) :
            PolicyDatasetExperiment(batch_size, chunk_size, max_epochs, trajectory_length, discount_factor, disable_balancing, true, false, sample_method),
            learning_rate_(learning_rate),
            disable_baseline_(disable_baseline),
            disable_value_regularization_(disable_value_regularization),
            sample_successor_(sample_successor) {};
    };
}  // namespace experiments

#endif  // EXPERIMENTS_POLICY_MODEL_ACTOR_CRITIC_HPP_
