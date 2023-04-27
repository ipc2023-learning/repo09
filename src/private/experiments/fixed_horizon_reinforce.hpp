#if !defined(EXPERIMENTS_REINFORCE_HPP_)
#define EXPERIMENTS_REINFORCE_HPP_

#include "../models/relational_neural_network.hpp"
#include "fixed_horizon_policy_gradient.hpp"
#include "torch/torch.h"
#include "values/optimal_value_function.hpp"
#include "values/value_functions.hpp"

#include <memory>
#include <unordered_map>

namespace experiments
{
    class FixedHorizonReinforce : public FixedHorizonPolicyGradient
    {
      private:
        torch::Tensor cumulative_return(int32_t step_index, const Trajectory& trajectory) const;

      protected:
        torch::Tensor get_update_value(const formalism::ProblemDescription& problem, const experiments::Trajectory& trajectory, std::size_t step_index) const;

      public:
        FixedHorizonReinforce(uint32_t batch_size,
                              uint32_t max_epochs,
                              double learning_rate,
                              uint32_t max_horizon,
                              double discount,
                              const RewardFunction& reward_function);
    };
}  // namespace experiments

#endif  // EXPERIMENTS_REINFORCE_HPP_
