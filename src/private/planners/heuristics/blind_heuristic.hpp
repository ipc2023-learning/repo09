#if !defined(PLANNERS_BLIND_HEURISTIC_HPP_)
#define PLANNERS_BLIND_HEURISTIC_HPP_

#include "../../formalism/atom.hpp"
#include "../../formalism/problem.hpp"
#include "../../formalism/state.hpp"
#include "heuristic_base.hpp"

#include <vector>

namespace planners
{
    class BlindHeuristic : public HeuristicBase
    {
      public:
        double get_cost(const formalism::State& state) const override;
    };
}  // namespace planners

#endif  // PLANNERS_BLIND_HEURISTIC_HPP_
