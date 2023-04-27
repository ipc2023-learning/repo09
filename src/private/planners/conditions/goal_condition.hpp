#if !defined(PLANNERS_GOAL_CONDITION_HPP_)
#define PLANNERS_GOAL_CONDITION_HPP_

#include "../../formalism/problem.hpp"
#include "condition_base.hpp"

namespace planners
{
    class GoalCondition : public ConditionBase
    {
      private:
        formalism::ProblemDescription problem_;

      public:
        GoalCondition(const formalism::ProblemDescription& problem);

        bool test(const formalism::State& state) override;
    };
}  // namespace planners

#endif  // PLANNERS_GOAL_CONDITION_HPP_
