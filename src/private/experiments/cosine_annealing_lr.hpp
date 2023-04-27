#if !defined(EXPERIMENTS_COSINE_ANNEALING_LR_HPP_)
#define EXPERIMENTS_COSINE_ANNEALING_LR_HPP_

#include "torch/torch.h"

namespace experiments
{
    class CosineAnnealingLR
    {
      private:
        torch::optim::Optimizer& optimizer_;
        size_t max_epochs_;
        double min_learning_rate_;

      public:
        using OnLearningRateUpdateCallback = std::function<void(double)>;

        CosineAnnealingLR(torch::optim::Optimizer& optimizer, size_t max_epochs, double min_learning_rate = 0.0);

        void step(size_t current_epoch);

        void on_lr_update(OnLearningRateUpdateCallback callback);

      private:
        OnLearningRateUpdateCallback callback_;
    };
}  // namespace experiments

#endif  // EXPERIMENTS_COSINE_ANNEALING_LR_HPP_
