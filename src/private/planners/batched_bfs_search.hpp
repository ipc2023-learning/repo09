#if !defined(PLANNERS_BATCHED_BFS_SEARCH_HPP_)
#define PLANNERS_BATCHED_BFS_SEARCH_HPP_

#include "../formalism/action.hpp"
#include "../formalism/declarations.hpp"
#include "../formalism/state.hpp"
#include "conditions/condition_base.hpp"
#include "generators/successor_generator.hpp"
#include "heuristics/heuristic_base.hpp"
#include "search_statistics.hpp"

namespace planners
{
    bool batched_bfs_search(const formalism::State& initial_state,
                            planners::SuccessorGenerator& successor_generator,
                            planners::HeuristicBase& heuristic,
                            planners::ConditionBase& goal,
                            planners::ConditionBase& prune,
                            SearchStatistics& statistics,
                            formalism::ActionList& out_plan);
}  // namespace planners

#endif  // PLANNERS_BATCHED_BFS_SEARCH_HPP_
