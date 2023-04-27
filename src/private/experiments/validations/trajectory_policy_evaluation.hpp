#if !defined(EXPERIMENTS_TRAJECTORY_POLICY_EVALUATION)
#define EXPERIMENTS_TRAJECTORY_POLICY_EVALUATION

#include "evaluation.hpp"
namespace experiments
{
    class TrajectoryPolicyEvaluation : public Evaluation
    {
      public:
        double initialize(const planners::StateSpaceList& state_spaces, models::RelationalNeuralNetwork& model) override;

        double evaluate(models::RelationalNeuralNetwork& model) override;
    };
}  // namespace experiments

#endif  // EXPERIMENTS_TRAJECTORY_POLICY_EVALUATION
