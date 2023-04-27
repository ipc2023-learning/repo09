#if !defined(PLANNERS_VALUE_SEARCH_HPP_)
#define PLANNERS_VALUE_SEARCH_HPP_

#include "../formalism/action.hpp"
#include "../formalism/domain.hpp"
#include "../formalism/problem.hpp"
#include "../models/relational_neural_network.hpp"

#include <chrono>
#include <vector>

namespace planners
{
    class ValueSearch
    {
      private:
        formalism::ProblemDescription problem;
        models::RelationalNeuralNetwork model;

      public:
        uint32_t generated;
        uint32_t expanded;

        int64_t time_total_ns;
        int64_t time_successors_ns;
        int64_t time_inference_ns;

        ValueSearch(const formalism::ProblemDescription& problem, const models::RelationalNeuralNetwork& model);

        bool find_plan(bool verbose, std::vector<formalism::Action>& plan);
    };
}  // namespace planners

#endif  // PLANNERS_VALUE_SEARCH_HPP_
