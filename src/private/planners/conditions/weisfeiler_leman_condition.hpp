#if !defined(PLANNERS_WEISFEILER_LEMAN_CONDITION_HPP_)
#define PLANNERS_WEISFEILER_LEMAN_CONDITION_HPP_

#include "../../algorithms/weisfeiler_leman.hpp"
#include "../../formalism/problem.hpp"
#include "condition_base.hpp"

#include <unordered_set>

namespace std
{
    template<>
    struct hash<pair<uint64_t, uint64_t>>
    {
        size_t operator()(const pair<uint64_t, uint64_t>& x) const { return (size_t) x.first; }
    };
}  // namespace std

namespace planners
{
    class WeisfeilerLemanCondition : public ConditionBase
    {
      private:
        formalism::ProblemDescription problem_;
        algorithms::WeisfeilerLeman weisfeiler_leman_;
        std::unordered_set<std::pair<uint64_t, uint64_t>> colors_;

      public:
        WeisfeilerLemanCondition(const formalism::ProblemDescription& problem);

        bool test(const formalism::State& state) override;
    };
}  // namespace planners

#endif  // PLANNERS_WEISFEILER_LEMAN_CONDITION_HPP_
