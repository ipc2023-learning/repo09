#if !defined(EXPERIMENTS_VALUE_SUPERVISED_OPTIMAL_HPP_)
#define EXPERIMENTS_VALUE_SUPERVISED_OPTIMAL_HPP_

#include "../models/relational_neural_network.hpp"
#include "dataset_experiment.hpp"
#include "torch/torch.h"

namespace experiments
{
    class SupervisedOptimal : public DatasetExperiment
    {
      private:
        torch::Tensor loss(const planners::StateSpaceSampleList& batch, const std::pair<torch::Tensor, torch::Tensor>& output);

      protected:
        torch::Tensor train_loss(const planners::StateSpaceSampleList& batch, const std::pair<torch::Tensor, torch::Tensor>& output) override;

        torch::Tensor validation_loss(const planners::StateSpaceSampleList& batch, const std::pair<torch::Tensor, torch::Tensor>& output) override;

      public:
        SupervisedOptimal(uint32_t batch_size, uint32_t chunk_size, uint32_t max_epochs) : DatasetExperiment(batch_size, chunk_size, max_epochs) {};
    };
}  // namespace experiments

#endif  // EXPERIMENTS_VALUE_SUPERVISED_OPTIMAL_HPP_
