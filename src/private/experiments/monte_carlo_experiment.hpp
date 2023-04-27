#if !defined(EXPERIMENTS_ONLINE_EXPERIMENT_HPP_)
#define EXPERIMENTS_ONLINE_EXPERIMENT_HPP_

#include "../formalism/declarations.hpp"
#include "../formalism/domain.hpp"
#include "../formalism/problem.hpp"
#include "../formalism/state.hpp"
#include "../models/relational_neural_network.hpp"
#include "../planners/state_space.hpp"
#include "rewards/reward_function.hpp"
#include "torch/torch.h"
#include "trajectory.hpp"

namespace experiments
{
    class MonteCarloExperiment
    {
      private:
        uint32_t batch_size_;
        uint32_t max_epochs_;
        uint32_t max_horizon_;
        RewardFunction reward_function_;

      protected:
        virtual std::shared_ptr<torch::optim::Optimizer> create_optimizer(models::RelationalNeuralNetwork& model) = 0;

        virtual std::pair<torch::Tensor, torch::Tensor> train_loss(const planners::StateSpaceSampleList& batch,
                                                                   const experiments::TrajectoryList& trajectories) = 0;

        virtual torch::Tensor validation_loss(const planners::StateSpaceSampleList& batch, const experiments::TrajectoryList& trajectories) = 0;

        experiments::TrajectoryList compute_trajectories(models::RelationalNeuralNetwork& model, const planners::StateSpaceSampleList& instances);

        RewardFunction get_reward_function() const;

      public:
        MonteCarloExperiment(uint32_t batch_size, uint32_t max_epochs, uint32_t max_horizon, const RewardFunction& reward_function);

        virtual ~MonteCarloExperiment();

        void fit(models::RelationalNeuralNetwork& model,
                 const planners::StateSpaceList& training_state_spaces,
                 const planners::StateSpaceList& validation_state_spaces);
    };

}  // namespace experiments

#endif  // EXPERIMENTS_ONLINE_EXPERIMENT_HPP_
