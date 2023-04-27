#if !defined(PLANNERS_HEURISTIC_BASE_HPP_)
#define PLANNERS_HEURISTIC_BASE_HPP_

#include "../../formalism/state.hpp"

#include <limits>
#include <vector>

namespace planners
{
    class HeuristicBase
    {
      public:
        static constexpr double DEAD_END = std::numeric_limits<double>::infinity();

        virtual double get_cost(const formalism::State& state) const = 0;

        virtual std::vector<double> get_cost(const std::vector<formalism::State>& states) const;
    };

}  // namespace planners

#endif  // PLANNERS_HEURISTIC_BASE_HPP_
