#if !defined(EXPERIMENTS_EXACT_POLICY_EVALUATION)
#define EXPERIMENTS_EXACT_POLICY_EVALUATION

#include "evaluation.hpp"

namespace experiments
{
    class ExactPolicyEvaluation : public Evaluation
    {
      private:
        double discount_factor_;
        double mean_optimal_value_;
        uint32_t batch_size_;
        planners::StateSpaceList state_spaces_;
        std::vector<TransitionProbabilities> validation_transition_probabilities_;
        std::vector<std::vector<formalism::StateTransitionsVector>> validation_batches_;
        std::vector<std::vector<std::vector<std::size_t>>> validation_indices_;

      public:
        ExactPolicyEvaluation(double discount_factor, uint32_t batch_size);

        double initialize(const planners::StateSpaceList& state_spaces, models::RelationalNeuralNetwork& model) override;

        double evaluate(models::RelationalNeuralNetwork& model) override;
    };
}  // namespace experiments

#endif  // EXPERIMENTS_EXACT_POLICY_EVALUATION
