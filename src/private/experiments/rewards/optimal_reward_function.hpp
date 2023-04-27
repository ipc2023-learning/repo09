#if !defined(EXPERIMENTS_OPTIMAL_REWARD_FUNCTION_HPP_)
#define EXPERIMENTS_OPTIMAL_REWARD_FUNCTION_HPP_

#include "../values/optimal_value_function.hpp"
#include "reward_function.hpp"

#include <memory>

namespace experiments
{
    class OptimalRewardFunctionImpl : public RewardFunctionImpl
    {
      private:
        OptimalValueFunction optimal_value_function_;

      public:
        OptimalRewardFunctionImpl();

        double cumulative_optimal_reward(const planners::StateSpace& state_space, const formalism::State& state);

        double reward(const planners::StateSpace& state_space, const formalism::State& from_state, const formalism::State& to_state);
    };

    using OptimalRewardFunction = std::shared_ptr<OptimalRewardFunctionImpl>;

    OptimalRewardFunction create_optimal_reward_function();
}  // namespace experiments

#endif  // EXPERIMENTS_OPTIMAL_REWARD_FUNCTION_HPP_
