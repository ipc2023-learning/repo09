#if !defined(EXPERIMENTS_VALUE_FUNCTIONS_HPP_)
#define EXPERIMENTS_VALUE_FUNCTIONS_HPP_

#include "../../formalism/state.hpp"
#include "../../planners/state_space.hpp"

namespace experiments
{
    class ValueFunction
    {
      public:
        virtual uint32_t get_value(const planners::StateSpace& state_space, const formalism::State& state) = 0;

        virtual ~ValueFunction();
    };
}  // namespace experiments

#endif  // EXPERIMENTS_VALUE_FUNCTIONS_HPP_
