#if !defined(EXPERIMENTS_FIXED_HORIZON_VALUE_BASE_HPP_)
#define EXPERIMENTS_FIXED_HORIZON_VALUE_BASE_HPP_

#include "fixed_horizon_policy_gradient.hpp"

namespace experiments
{
    class FixedHorizonValueBase : public FixedHorizonPolicyGradient
    {
      private:
        double bounds_factor_;

      protected:
        double get_bounds_factor() const;

        torch::Tensor update_value_gradient(const torch::Tensor gradient,
                                            const torch::Tensor update_value,
                                            const experiments::TrajectoryStep& step,
                                            const double cumulative_discount) const;

      public:
        FixedHorizonValueBase(uint32_t batch_size,
                              uint32_t max_epochs,
                              double learning_rate,
                              uint32_t max_horizon,
                              double discount,
                              double bounds_factor,
                              const RewardFunction& reward_function);
    };
}  // namespace experiments

#endif  // EXPERIMENTS_FIXED_HORIZON_VALUE_BASE_HPP_
