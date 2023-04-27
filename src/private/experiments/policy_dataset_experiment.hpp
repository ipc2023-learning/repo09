#if !defined(EXPERIMENTS_POLICY_DATASET_EXPERIMENT_HPP_)
#define EXPERIMENTS_POLICY_DATASET_EXPERIMENT_HPP_

#include "../datasets/dataset.hpp"
#include "../formalism/declarations.hpp"
#include "../models/relational_neural_network.hpp"
#include "../planners/state_space.hpp"
#include "torch/torch.h"

#include <chrono>
#include <map>

namespace experiments
{
    enum PolicySamplingMethod
    {
        Policy,
        Uniform
    };

    struct PolicyBatchOutput
    {
        std::vector<std::pair<formalism::State, formalism::StateList>> state_successors;
        std::vector<torch::Tensor> policies;
        std::vector<torch::Tensor> values;
        std::vector<torch::Tensor> dead_ends;
        std::vector<int64_t> sampled_successor_indices;
        std::vector<torch::Tensor> sampled_successor_probabilities;
        std::vector<torch::Tensor> sampled_successor_values;
        std::vector<torch::Tensor> sampled_successor_dead_ends;
        std::vector<formalism::State> sampled_successor_states;
        int32_t step;

        PolicyBatchOutput(const std::vector<std::pair<formalism::State, formalism::StateList>>& state_successors,
                          const std::vector<torch::Tensor>& policies,
                          const std::vector<torch::Tensor>& values,
                          const std::vector<torch::Tensor>& dead_ends,
                          const std::vector<int64_t>& sampled_successor_indices,
                          const std::vector<torch::Tensor>& sampled_successor_probabilities,
                          const std::vector<torch::Tensor>& sampled_successor_values,
                          const std::vector<torch::Tensor>& sampled_successor_dead_ends,
                          const std::vector<formalism::State>& sampled_successor_states,
                          const int32_t step) :
            state_successors(state_successors),
            policies(policies),
            values(values),
            dead_ends(dead_ends),
            sampled_successor_indices(sampled_successor_indices),
            sampled_successor_probabilities(sampled_successor_probabilities),
            sampled_successor_values(sampled_successor_values),
            sampled_successor_dead_ends(sampled_successor_dead_ends),
            sampled_successor_states(sampled_successor_states),
            step(step)
        {
        }
    };

    // TODO: It's better to rename this to Temporal Difference Experiment since updates are done every step.

    class PolicyDatasetExperiment
    {
      private:
        std::map<formalism::ProblemDescription, planners::StateSpace> state_spaces_;
        uint32_t batch_size_;
        uint32_t chunk_size_;
        uint32_t max_epochs_;
        uint32_t trajectory_length_;
        double discount_factor_;
        bool disable_balancing_;
        bool remove_goal_states_;
        bool remove_dead_end_states_;
        PolicySamplingMethod sampling_method_;

        void log_timings(const std::string& message,
                         std::chrono::high_resolution_clock::time_point inner_start,
                         std::chrono::high_resolution_clock::time_point outer_start,
                         uint32_t epoch);

        void log_timings(const std::string& message,
                         std::chrono::high_resolution_clock::time_point inner_start,
                         std::chrono::high_resolution_clock::time_point outer_start,
                         uint32_t epoch,
                         uint32_t index,
                         uint32_t length);

        void log_timings(const std::string& message,
                         std::chrono::high_resolution_clock::time_point inner_start,
                         std::chrono::high_resolution_clock::time_point outer_start,
                         uint32_t epoch,
                         uint32_t outer_index,
                         uint32_t outer_length,
                         uint32_t inner_index,
                         uint32_t inner_length,
                         double value);

        void log_timings(const std::string& message,
                         std::chrono::high_resolution_clock::time_point inner_start,
                         std::chrono::high_resolution_clock::time_point outer_start,
                         uint32_t epoch,
                         double value);

        void log_and_reset_divergence(std::map<planners::StateSpace, std::vector<uint32_t>>& distributions,
                                      uint64_t num_states,
                                      uint64_t num_samples,
                                      std::chrono::high_resolution_clock::time_point inner_start,
                                      std::chrono::high_resolution_clock::time_point outer_start,
                                      uint32_t epoch);

      protected:
        virtual std::shared_ptr<torch::optim::Optimizer> create_optimizer(models::RelationalNeuralNetwork& model) = 0;

        virtual torch::Tensor train_loss(uint32_t epoch, const planners::StateSpaceSampleList& batch, const PolicyBatchOutput& output);

        virtual torch::Tensor
        train_loss(uint32_t epoch, const planners::StateSpaceSampleList& batch, const PolicyBatchOutput& output, models::RelationalNeuralNetwork& model);

        virtual planners::StateSpaceSampleList
        get_batch(const std::shared_ptr<datasets::Dataset>& set, uint32_t epoch, uint32_t batch_index, uint32_t batch_size);

        uint32_t get_batch_size() const;

        double get_discount_factor() const;

      public:
        PolicyDatasetExperiment(uint32_t batch_size,
                                uint32_t chunk_size,
                                uint32_t max_epochs,
                                uint32_t trajectory_length,
                                double discount_factor,
                                bool disable_balancing,
                                bool remove_goal_states,
                                bool remove_dead_end_states,
                                PolicySamplingMethod sampling_method) :
            batch_size_(batch_size),
            chunk_size_(chunk_size),
            max_epochs_(max_epochs),
            trajectory_length_(trajectory_length > 1 ? trajectory_length : 1),
            discount_factor_(discount_factor),
            disable_balancing_(disable_balancing),
            remove_goal_states_(remove_goal_states),
            remove_dead_end_states_(remove_dead_end_states),
            sampling_method_(sampling_method)
        {
        }

        virtual ~PolicyDatasetExperiment() {}

        virtual void fit(models::RelationalNeuralNetwork& model, const planners::StateSpaceList& training_set, const planners::StateSpaceList& validation_set);
    };

}  // namespace experiments

#endif  // EXPERIMENTS_POLICY_DATASET_EXPERIMENT_HPP_
