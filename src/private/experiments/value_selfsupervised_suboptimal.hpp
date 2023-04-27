#if !defined(EXPERIMENTS_VALUE_SELFSUPERVISED_SUBOPTIMAL_HPP_)
#define EXPERIMENTS_VALUE_SELFSUPERVISED_SUBOPTIMAL_HPP_

#include "../models/relational_neural_network.hpp"
#include "dataset_experiment.hpp"
#include "torch/torch.h"

namespace experiments
{
    class SelfsupervisedSuboptimal : public DatasetExperiment
    {
      private:
        double approximation_factor_;

        torch::Tensor loss(const planners::StateSpaceSampleList& batch, const std::pair<torch::Tensor, torch::Tensor>& output);

      protected:
        torch::Tensor train_loss(const planners::StateSpaceSampleList& batch, const std::pair<torch::Tensor, torch::Tensor>& output) override;

        torch::Tensor validation_loss(const planners::StateSpaceSampleList& batch, const std::pair<torch::Tensor, torch::Tensor>& output) override;

        planners::StateSpaceSampleList get_batch(const datasets::Dataset& set, uint32_t batch_index, uint32_t batch_size) override;

      public:
        SelfsupervisedSuboptimal(uint32_t batch_size, uint32_t chunk_size, uint32_t max_epochs, double approximation_factor) :
            DatasetExperiment(batch_size, chunk_size, max_epochs),
            approximation_factor_(approximation_factor) {};
    };
}  // namespace experiments

#endif  // EXPERIMENTS_VALUE_SELFSUPERVISED_SUBOPTIMAL_HPP_
