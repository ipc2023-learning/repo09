#if !defined(PLANNERS_ONE_NOVELTY_CONDITION_HPP_)
#define PLANNERS_ONE_NOVELTY_CONDITION_HPP_

#include "../../formalism/problem.hpp"
#include "condition_base.hpp"

namespace planners
{
    class OneNoveltyCondition : public ConditionBase
    {
      private:
        formalism::ProblemDescription problem_;
        std::vector<bool> seen_;

      public:
        OneNoveltyCondition(const formalism::ProblemDescription& problem);

        bool test(const formalism::State& state) override;
    };
}  // namespace planners

#endif  // PLANNERS_ONE_NOVELTY_CONDITION_HPP_
