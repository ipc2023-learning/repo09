#if !defined(PLANNERS_SCALE_HEURISTIC_HPP_)
#define PLANNERS_SCALE_HEURISTIC_HPP_

#include "../../formalism/atom.hpp"
#include "../../formalism/problem.hpp"
#include "../../formalism/state.hpp"
#include "heuristic_base.hpp"

#include <vector>

namespace planners
{
    class ScaleHeuristic : public HeuristicBase
    {
      private:
        double scalar_;
        HeuristicBase& heuristic_;

      public:
        ScaleHeuristic(double scalar, HeuristicBase& heuristic);

        double get_cost(const formalism::State& state) const override;

        std::vector<double> get_cost(const std::vector<formalism::State>& states) const override;
    };
}  // namespace planners

#endif  // PLANNERS_SCALE_HEURISTIC_HPP_
