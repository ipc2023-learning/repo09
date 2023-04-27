// #if !defined(PLANNERS_BFS_SEARCH_HPP_)
// #define PLANNERS_BFS_SEARCH_HPP_

// #include "../formalism/action.hpp"
// #include "../formalism/domain.hpp"
// #include "../formalism/problem.hpp"
// #include "../formalism/state.hpp"
// #include "../models/relational_neural_network.hpp"
// #include "generators/successor_generator_factory.hpp"
// #include "search_base.hpp"

// #include <vector>

// namespace planners
// {
//     class AStarSearch : SearchBase<void>
//     {
//       public:
//         uint32_t generated;
//         uint32_t expanded;
//         uint32_t max_expanded;
//         int64_t time_total_ns;
//         int64_t time_successors_ns;

//         AStarSearch();

//         std::vector<int32_t> heuristic(const std::vector<formalism::State>& states);

//         bool find_plan(const formalism::ProblemDescription& problem,
//                        planners::SuccessorGenerator& successor_generator,
//                        planners::HeuristicBase& heuristic,
//                        planners::ConditionBase& goal,
//                        planners::ConditionBase& prune,
//                        std::vector<formalism::Action>& out_plan) override;

//         void set_max_expanded(uint32_t max);
//     };
// }  // namespace planners

// #endif  // PLANNERS_BFS_SEARCH_HPP_
