#if !defined(EXPERIMENTS_POLICY_TABULAR_ITERATIVE_PROBABILITY_HPP_)
#define EXPERIMENTS_POLICY_TABULAR_ITERATIVE_PROBABILITY_HPP_

#include "../models/relational_neural_network.hpp"
#include "../planners/state_space.hpp"
#include "policy_dataset_experiment.hpp"
#include "torch/torch.h"

#include <map>
#include <vector>

namespace experiments
{
    class PolicyIterativeTabularProbability : public PolicyDatasetExperiment
    {
      private:
        double learning_rate_;
        bool disable_baseline_;
        bool disable_value_regularization_;
        std::map<formalism::ProblemDescription, std::vector<double>> problem_optimal_vector_;
        std::map<formalism::ProblemDescription, std::vector<double>> problem_value_vector_;
        std::map<formalism::ProblemDescription, std::vector<double>> problem_probability_vector_;

        void initialize_tables_if_necessary(const planners::StateSpace& state_space, models::RelationalNeuralNetwork& model);

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
        PolicyIterativeTabularProbability(uint32_t batch_size,
                                          uint32_t chunk_size,
                                          uint32_t max_epochs,
                                          double learning_rate,
                                          double discount_factor,
                                          bool disable_balancing,
                                          bool disable_baseline,
                                          bool disable_value_regularization);
    };
}  // namespace experiments

#endif  // EXPERIMENTS_POLICY_TABULAR_ITERATIVE_PROBABILITY_HPP_
