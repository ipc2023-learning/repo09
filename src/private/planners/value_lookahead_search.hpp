#if !defined(PLANNERS_VALUE_LOOKAHEAD_SEARCH_HPP_)
#define PLANNERS_VALUE_LOOKAHEAD_SEARCH_HPP_

#include "../formalism/action.hpp"
#include "../formalism/domain.hpp"
#include "../formalism/problem.hpp"
#include "../models/relational_neural_network.hpp"
#include "generators/successor_generator.hpp"

#include <chrono>
#include <vector>

namespace planners
{
    class ValueLookaheadSearch
    {
      private:
        struct LookaheadResult;

        formalism::ProblemDescription problem_;
        models::RelationalNeuralNetwork model_;
        bool use_weisfeiler_leman_;
        uint32_t chunk_size_;

        std::vector<LookaheadResult> lookahead_search(const formalism::State& initial_state,
                                                      double initial_state_value,
                                                      double initial_state_dead_end,
                                                      double value_difference_threshold,
                                                      const planners::SuccessorGenerator& successor_generator);

      public:
        uint32_t generated;
        uint32_t expanded;

        int64_t time_total_ns;
        int64_t time_successors_ns;
        int64_t time_inference_ns;

        ValueLookaheadSearch(const formalism::ProblemDescription& problem,
                             const models::RelationalNeuralNetwork& model,
                             bool use_weisfeiler_leman,
                             uint32_t chunk_size);

        bool find_plan(bool verbose, std::vector<formalism::Action>& plan_actions, std::vector<double>& plan_values, std::vector<double>& plan_dead_ends);
    };
}  // namespace planners

#endif  // PLANNERS_VALUE_LOOKAHEAD_SEARCH_HPP_
