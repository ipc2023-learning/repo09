#if !defined(EXPERIMENTS_TABULAR_MODEL_ACTOR_CRITIC_HPP_)
#define EXPERIMENTS_TABULAR_MODEL_ACTOR_CRITIC_HPP_

#include "../models/relational_neural_network.hpp"
#include "../planners/state_space.hpp"
#include "policy_dataset_experiment.hpp"
#include "torch/torch.h"

#include <map>
#include <vector>

namespace experiments
{
    class PolicyTabularModelActorCritic : public PolicyDatasetExperiment
    {
      private:
        double learning_rate_;
        bool disable_baseline_;
        std::map<formalism::ProblemDescription, std::vector<double>> problem_value_vector_;
        std::map<formalism::ProblemDescription, std::vector<std::vector<std::pair<uint64_t, double>>>> problem_transitions_vector_;

        void initialize_probabilities_and_values(const planners::StateSpaceSampleList& batch, models::RelationalNeuralNetwork& model);

        void update_transition_probabilities(const planners::StateSpaceSampleList& batch, const PolicyBatchOutput& output);

        void update_values(const planners::StateSpaceSampleList& batch);

        torch::Tensor loss(const planners::StateSpaceSampleList& batch, const PolicyBatchOutput& output, models::RelationalNeuralNetwork& model);

      protected:
        std::shared_ptr<torch::optim::Optimizer> create_optimizer(models::RelationalNeuralNetwork& model) override;

        torch::Tensor train_loss(uint32_t epoch,
                                 const planners::StateSpaceSampleList& batch,
                                 const PolicyBatchOutput& output,
                                 models::RelationalNeuralNetwork& model) override;

        planners::StateSpaceSampleList
        get_batch(const std::shared_ptr<datasets::Dataset>& set, uint32_t epoch, uint32_t batch_index, uint32_t batch_size) override;

      public:
        PolicyTabularModelActorCritic(uint32_t batch_size,
                                      uint32_t chunk_size,
                                      uint32_t max_epochs,
                                      uint32_t trajectory_length,
                                      double learning_rate,
                                      double discount_factor,
                                      bool disable_balancing,
                                      bool disable_baseline);
    };
}  // namespace experiments

#endif  // EXPERIMENTS_TABULAR_MODEL_ACTOR_CRITIC_HPP_
