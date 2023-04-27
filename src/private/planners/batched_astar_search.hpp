#if !defined(PLANNERS_BATCHED_ASTAR_SEARCH_HPP_)
#define PLANNERS_BATCHED_ASTAR_SEARCH_HPP_

#include "../formalism/action.hpp"
#include "../formalism/declarations.hpp"
#include "../formalism/state.hpp"
#include "conditions/condition_base.hpp"
#include "generators/successor_generator.hpp"
#include "heuristics/heuristic_base.hpp"
#include "search_base.hpp"
#include "search_statistics.hpp"

#include <chrono>

namespace planners
{
    struct BatchedAstarSettings
    {
        double batch_delta;
        double min_improvement;
        uint32_t batch_size;
        uint32_t min_expanded;
        uint32_t max_expanded;
    };

    bool batched_astar_search(const BatchedAstarSettings& settings,
                              const formalism::State& initial_state,
                              planners::SearchStateRepository& state_repository,
                              planners::SuccessorGenerator& successor_generator,
                              planners::HeuristicBase& heuristic,
                              planners::ConditionBase& goal,
                              planners::ConditionBase& prune,
                              SearchStatistics& statistics,
                              std::chrono::high_resolution_clock::time_point& time_end,
                              formalism::ActionList& out_plan);
}  // namespace planners

#endif  // PLANNERS_BATCHED_ASTAR_SEARCH_HPP_
