#if !defined(PLANNERS_TWO_NOVELTY_CONDITION_HPP_)
#define PLANNERS_TWO_NOVELTY_CONDITION_HPP_

#include "../../formalism/problem.hpp"
#include "condition_base.hpp"

#include <unordered_set>

namespace std
{
    template<>
    struct hash<pair<uint32_t, uint32_t>>
    {
        size_t operator()(const pair<uint32_t, uint32_t>& x) const { return (static_cast<size_t>(x.first) << 32) | static_cast<size_t>(x.second); }
    };
}  // namespace std

namespace planners
{
    class TwoNoveltyCondition : public ConditionBase
    {
      private:
        formalism::ProblemDescription problem_;
        std::unordered_set<std::pair<uint32_t, uint32_t>> seen_;

      public:
        TwoNoveltyCondition(const formalism::ProblemDescription& problem);

        bool test(const formalism::State& state) override;
    };
}  // namespace planners

#endif  // PLANNERS_TWO_NOVELTY_CONDITION_HPP_
