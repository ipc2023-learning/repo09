#if !defined(EXPERIMENTS_DATASET_EXPERIMENT_HPP_)
#define EXPERIMENTS_DATASET_EXPERIMENT_HPP_

#include "../datasets/dataset.hpp"
#include "../formalism/declarations.hpp"
#include "../models/relational_neural_network.hpp"
#include "../planners/state_space.hpp"
#include "torch/torch.h"

#include <map>

namespace experiments
{
    struct TrainingStep
    {
        uint32_t epoch;
        uint32_t batch_index;
        uint32_t num_batches;
        uint32_t time_ms;
    };

    struct ValidationStep
    {
        uint32_t epoch;
        uint32_t time_ms;
    };

    struct EpochStep
    {
        uint32_t epoch;
        uint32_t time_ms;
    };

    class DatasetExperiment
    {
      public:
        using OnTrainingStepCallback = std::function<void(const TrainingStep&, double)>;
        using OnValidationStepCallback = std::function<void(const ValidationStep&, double)>;
        using OnEpochStepCallback = std::function<bool(const EpochStep&)>;

      private:
        std::map<formalism::ProblemDescription, planners::StateSpace> state_spaces_;
        uint32_t batch_size;
        uint32_t chunk_size;
        uint32_t max_epochs;

        OnTrainingStepCallback training_callback_;
        OnValidationStepCallback validation_callback_;
        OnEpochStepCallback epoch_callback_;

      protected:
        virtual torch::Tensor train_loss(const planners::StateSpaceSampleList& batch, const std::pair<torch::Tensor, torch::Tensor>& output) = 0;

        virtual torch::Tensor validation_loss(const planners::StateSpaceSampleList& batch, const std::pair<torch::Tensor, torch::Tensor>& output) = 0;

        virtual planners::StateSpaceSampleList get_batch(const datasets::Dataset& set, uint32_t batch_index, uint32_t batch_size);

      public:
        DatasetExperiment(uint32_t batch_size, uint32_t chunk_size, uint32_t max_epochs) :
            batch_size(batch_size),
            chunk_size(chunk_size),
            max_epochs(max_epochs)
        {
        }

        virtual ~DatasetExperiment() {}

        virtual void fit(models::RelationalNeuralNetwork& model,
                         torch::optim::Optimizer& optimizer,
                         const datasets::Dataset& training_set,
                         const datasets::Dataset& validation_set);

        void register_on_training_step(const OnTrainingStepCallback& callback);
        void register_on_validation_step(const OnValidationStepCallback& callback);
        void register_on_epoch_step(const OnEpochStepCallback& callback);
    };

}  // namespace experiments

#endif  // EXPERIMENTS_DATASET_EXPERIMENT_HPP_
