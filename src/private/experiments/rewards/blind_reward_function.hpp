#if !defined(EXPERIMENTS_BLIND_REWARD_FUNCTION_HPP_)
#define EXPERIMENTS_BLIND_REWARD_FUNCTION_HPP_

#include "../values/optimal_value_function.hpp"
#include "reward_function.hpp"

#include <memory>

namespace experiments
{
    class BlindRewardFunctionImpl : public RewardFunctionImpl
    {
      private:
        OptimalValueFunction optimal_value_function_;

      public:
        BlindRewardFunctionImpl();

        double cumulative_optimal_reward(const planners::StateSpace& state_space, const formalism::State& state);

        double reward(const planners::StateSpace& state_space, const formalism::State& from_state, const formalism::State& to_state);
    };

    using BlindRewardFunction = std::shared_ptr<BlindRewardFunctionImpl>;

    BlindRewardFunction create_blind_reward_function();
}  // namespace experiments

#endif  // EXPERIMENTS_BLIND_REWARD_FUNCTION_HPP_
