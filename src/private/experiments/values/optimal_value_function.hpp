#if !defined(EXPERIMENTS_OPTIMAL_VALUE_FUNCTION_HPP_)
#define EXPERIMENTS_OPTIMAL_VALUE_FUNCTION_HPP_

#include "../../formalism/problem.hpp"
#include "../../formalism/state.hpp"
#include "value_functions.hpp"

#include <unordered_map>

namespace experiments
{
    class OptimalValueFunction : public ValueFunction
    {
      public:
        OptimalValueFunction();

        uint32_t get_value(const planners::StateSpace& state_space, const formalism::State& state);
    };
}  // namespace experiments

#endif  // EXPERIMENTS_OPTIMAL_VALUE_FUNCTION_HPP_
