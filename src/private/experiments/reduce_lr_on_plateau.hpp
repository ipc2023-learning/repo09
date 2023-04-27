#if !defined(EXPERIMENTS_REDUCE_LR_ON_PLATEAU_HPP_)
#define EXPERIMENTS_REDUCE_LR_ON_PLATEAU_HPP_

#include "torch/torch.h"

namespace experiments
{
    class ReduceLROnPlateau
    {
      private:
        torch::optim::Optimizer& optimizer_;
        double factor_;
        double best_loss_;
        double min_learning_rate_;
        uint32_t patience_;
        uint32_t plateau_counter_;

        void decrease_learning_rate();

      public:
        using CallbackType = std::function<void(double)>;

        ReduceLROnPlateau(torch::optim::Optimizer& optimizer, double factor = 0.1, uint32_t patience = 10, double min_learning_rate = 0.0);

        void update(double loss);

        void register_on_lr_update(CallbackType callback);

      private:
        CallbackType callback_;
    };
}  // namespace experiments

#endif  // EXPERIMENTS_REDUCE_LR_ON_PLATEAU_HPP_
