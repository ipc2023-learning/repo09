#if !defined(PLANNERS_CONDITION_BASE_HPP_)
#define PLANNERS_CONDITION_BASE_HPP_

#include "../../formalism/state.hpp"

#include <vector>

namespace planners
{
    class ConditionBase
    {
      public:
        virtual bool test(const formalism::State& state) = 0;

        virtual std::vector<bool> test(const std::vector<formalism::State>& states);
    };

}  // namespace planners

#endif  // PLANNERS_CONDITION_BASE_HPP_
