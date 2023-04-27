#if !defined(EXPERIMENTS_VANILLA_POLICY_GRADIENT_HPP_)
#define EXPERIMENTS_VANILLA_POLICY_GRADIENT_HPP_

#include "../formalism/declarations.hpp"
#include "../models/relational_neural_network.hpp"
#include "monte_carlo_experiment.hpp"
#include "torch/torch.h"
#include "values/optimal_value_function.hpp"
#include "values/value_functions.hpp"

#include <memory>
#include <unordered_map>

namespace experiments
{
    class FixedHorizonPolicyGradient : public MonteCarloExperiment
    {
      private:
        double learning_rate_;
        double discount_;

        std::pair<torch::Tensor, torch::Tensor>
        loss(const planners::StateSpaceSampleList& batch, const experiments::TrajectoryList& trajectories, bool compute_gradient);

      protected:
        double get_learning_rate() const;

        double get_discount() const;

        virtual torch::Tensor
        get_update_value(const formalism::ProblemDescription& problem, const experiments::Trajectory& trajectory, std::size_t step_index) const = 0;

        virtual torch::Tensor update_transition_gradient(const torch::Tensor gradient,
                                                         const torch::Tensor update_value,
                                                         const experiments::TrajectoryStep& step,
                                                         const double cumulative_discount) const;

        virtual torch::Tensor update_value_gradient(const torch::Tensor gradient,
                                                    const torch::Tensor update_value,
                                                    const experiments::TrajectoryStep& step,
                                                    const double cumulative_discount) const;

      public:
        FixedHorizonPolicyGradient(uint32_t batch_size,
                                   uint32_t max_epochs,
                                   double learning_rate,
                                   uint32_t max_horizon,
                                   double discount,
                                   const RewardFunction& reward_function);

        virtual ~FixedHorizonPolicyGradient();

        std::shared_ptr<torch::optim::Optimizer> create_optimizer(models::RelationalNeuralNetwork& model);

        std::pair<torch::Tensor, torch::Tensor> train_loss(const planners::StateSpaceSampleList& batch, const experiments::TrajectoryList& trajectories);

        torch::Tensor validation_loss(const planners::StateSpaceSampleList& batch, const experiments::TrajectoryList& trajectories);
    };
}  // namespace experiments

#endif  // EXPERIMENTS_VANILLA_POLICY_GRADIENT_HPP_
