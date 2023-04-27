#if !defined(EXPERIMENTS_REWARD_FUNCTION_HPP_)
#define EXPERIMENTS_REWARD_FUNCTION_HPP_

#include "../../formalism/problem.hpp"
#include "../../formalism/state.hpp"

#include <memory>

namespace experiments
{
    class RewardFunctionImpl
    {
      public:
        /**
         * @brief Maximum reward that can be achieved from the given state.
         *
         * @param problem The problem the given state is associated with.
         * @param state A state.
         * @return double Maximum reward possible from the given state.
         */
        virtual double cumulative_optimal_reward(const planners::StateSpace& state_space, const formalism::State& state) = 0;

        /**
         * @brief Reward associated with the transition.
         *
         * @param problem The problem the given states are associated with.
         * @param from_state A source state.
         * @param to_state A target state.
         * @return double A reward associated with going from the source state to the target state.
         */
        virtual double reward(const planners::StateSpace& state_space, const formalism::State& from_state, const formalism::State& to_state) = 0;

        virtual ~RewardFunctionImpl() {};
    };

    using RewardFunction = std::shared_ptr<RewardFunctionImpl>;
}  // namespace experiments

#endif  // EXPERIMENTS_REWARD_FUNCTION_HPP_
