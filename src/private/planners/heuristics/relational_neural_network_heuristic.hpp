#if !defined(PLANNERS_RELATIONAL_NEURAL_NETWORK_HEURISTIC_HPP_)
#define PLANNERS_RELATIONAL_NEURAL_NETWORK_HEURISTIC_HPP_

#include "../../formalism/atom.hpp"
#include "../../formalism/problem.hpp"
#include "../../formalism/state.hpp"
#include "../../models/relational_neural_network.hpp"
#include "heuristic_base.hpp"

#include <unordered_map>
#include <vector>

namespace planners
{
    class RelationalNeuralNetworkHeuristic : public HeuristicBase
    {
      private:
        formalism::ProblemDescription problem_;
        mutable models::RelationalNeuralNetwork model_;
        mutable std::unordered_map<formalism::State, double> cache_;
        uint32_t chunk_size_;

      public:
        RelationalNeuralNetworkHeuristic(const formalism::ProblemDescription& problem, const models::RelationalNeuralNetwork& model, uint32_t chunk_size);

        double get_cost(const formalism::State& state) const override;

        std::vector<double> get_cost(const std::vector<formalism::State>& states) const override;
    };
}  // namespace planners

#endif  // PLANNERS_RELATIONAL_NEURAL_NETWORK_HEURISTIC_HPP_
