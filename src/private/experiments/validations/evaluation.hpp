#if !defined(EXPERIMENTS_EVALUATIOH_HPP)
#define EXPERIMENTS_EVALUATIOH_HPP

#include "../../models/relational_neural_network.hpp"
#include "../../planners/state_space.hpp"

namespace experiments
{
    class Evaluation
    {
      public:
        virtual double initialize(const planners::StateSpaceList& state_spaces, models::RelationalNeuralNetwork& model) = 0;

        virtual double evaluate(models::RelationalNeuralNetwork& model) = 0;
    };
}  // namespace experiments

#endif  // EXPERIMENTS_EVALUATIOH_HPP
