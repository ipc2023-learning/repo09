#if !defined(PLANNERS_VALUE_DECOMPOSITION_SEARCH_HPP_)
#define PLANNERS_VALUE_DECOMPOSITION_SEARCH_HPP_

#include "../formalism/declarations.hpp"
#include "../formalism/problem.hpp"

#include <functional>

namespace planners
{
    bool decomposition_search(const formalism::ProblemDescription& problem,
                              const std::function<bool(const formalism::State&, formalism::ActionList&)>& planner,
                              formalism::ActionList& out_plan);
}  // namespace planners

#endif  // PLANNERS_VALUE_DECOMPOSITION_SEARCH_HPP_
