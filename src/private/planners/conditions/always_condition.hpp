#if !defined(PLANNERS_ALWAYS_CONDITION_HPP_)
#define PLANNERS_ALWAYS_CONDITION_HPP_

#include "../../formalism/problem.hpp"
#include "condition_base.hpp"

namespace planners
{
    class AlwaysCondition : public ConditionBase
    {
      private:
        bool boolean_;

      public:
        AlwaysCondition(bool boolean);

        bool test(const formalism::State& state) override;
    };
}  // namespace planners

#endif  // PLANNERS_ALWAYS_CONDITION_HPP_
