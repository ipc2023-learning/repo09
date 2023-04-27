#if !defined(EXPERIMENTS_IMAGE_SUPERVISED_OPTIMAL_HPP_)
#define EXPERIMENTS_IMAGE_SUPERVISED_OPTIMAL_HPP_

#include "../models/relational_neural_network.hpp"
#include "dataset_experiment.hpp"
#include "torch/torch.h"

namespace experiments
{
    class ImageSupervisedOptimal
    {
      private:
        uint32_t batch_size_;
        uint32_t max_epochs_;
        double learning_rate_;

      public:
        ImageSupervisedOptimal(uint32_t batch_size, uint32_t max_epochs, double learning_rate) :
            batch_size_(batch_size),
            max_epochs_(max_epochs),
            learning_rate_(learning_rate) {};

        void fit(models::RelationalNeuralNetwork& model,
                 const planners::StateSpaceList& training_state_spaces,
                 const planners::StateSpaceList& validation_state_spaces);
    };
}  // namespace experiments

#endif  // EXPERIMENTS_IMAGE_SUPERVISED_OPTIMAL_HPP_
