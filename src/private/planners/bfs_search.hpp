#if !defined(PLANNERS_BFS_SEARCH_HPP_)
#define PLANNERS_BFS_SEARCH_HPP_

#include "../formalism/action.hpp"
#include "../formalism/domain.hpp"
#include "../formalism/problem.hpp"
#include "generators/successor_generator_factory.hpp"

#include <vector>

namespace planners
{
    class BreadthFirstSearch
    {
      private:
        formalism::ProblemDescription problem_;
        planners::SuccessorGeneratorType successor_generator_type_;
        bool print_;

      public:
        uint32_t generated;
        uint32_t expanded;
        uint32_t max_expanded;

        int64_t time_total_ns;
        int64_t time_successors_ns;

        BreadthFirstSearch(const formalism::ProblemDescription& problem, planners::SuccessorGeneratorType successor_generator_type);

        bool find_plan(std::vector<formalism::Action>& plan);

        void print_progress(bool flag);

        void set_max_expanded(uint32_t max);
    };
}  // namespace planners

#endif  // PLANNERS_BFS_SEARCH_HPP_
