#if !defined(PLANNERS_SEARCH_BASE_HPP_)
#define PLANNERS_SEARCH_BASE_HPP_

#include "../formalism/action.hpp"
#include "../formalism/declarations.hpp"
#include "../formalism/state.hpp"
#include "conditions/condition_base.hpp"
#include "generators/successor_generator.hpp"
#include "heuristics/heuristic_base.hpp"
#include "search_statistics.hpp"

#include <deque>

namespace planners
{
    struct StateContext
    {
        double cost;
        double heuristic_value;
        formalism::Action predecessor_action;
        uint32_t predecessor_state_index;
    };

    class SearchStateRepository
    {
      private:
        std::deque<formalism::State> states_;
        std::unordered_map<formalism::State, uint32_t> indices_;
        std::deque<StateContext> contexts_;

      public:
        SearchStateRepository();

        bool add_or_get_state(const formalism::State& state, uint32_t& out_state_index);

        StateContext& get_context(uint32_t state_index);

        formalism::State& get_state(uint32_t state_index);

        uint32_t num_indices() const;

        std::vector<formalism::Action> get_plan(uint32_t state_index);
    };

    class SearchBase
    {
      protected:
        SearchStateRepository repository;

      public:
        SearchBase();

        virtual bool find_plan(const formalism::State& initial_state,
                               planners::SuccessorGenerator& successor_generator,
                               planners::HeuristicBase& heuristic,
                               planners::ConditionBase& goal,
                               planners::ConditionBase& prune,
                               SearchStatistics& statistics,
                               std::vector<formalism::Action>& out_plan) = 0;
    };
}  // namespace planners

#endif  // PLANNERS_SEARCH_BASE_HPP_
