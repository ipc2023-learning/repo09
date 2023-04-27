#if !defined(RELATIONAL_MESSAGE_PASSING_NEURAL_NETWORK_HPP_)
#define RELATIONAL_MESSAGE_PASSING_NEURAL_NETWORK_HPP_

#include "../formalism/atom.hpp"
#include "../formalism/declarations.hpp"
#include "../formalism/state.hpp"
#include "modules/readout_module.hpp"
#include "modules/relational_message_passing_module.hpp"
#include "relational_neural_network_base.hpp"
#include "torch/torch.h"

#include <string>
#include <vector>

// Older versions of LibC++ does not have filesystem (e.g., ubuntu 18.04), use the experimental version
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

namespace models
{
    class RelationalMessagePassingNeuralNetworkImpl : public RelationalNeuralNetworkBase
    {
      private:
        int32_t hidden_size_;
        int32_t num_layers_;
        bool global_readout_;
        double maximum_smoothness_;
        RelationalMessagePassingModule message_passing_module_;
        MLPModule global_module_;
        MLPModule transition_module_;
        ReadoutModule readout_global_module_;
        ReadoutModule readout_transition_module_;
        ReadoutModule readout_value_module_;
        ReadoutModule readout_dead_end_module_;

      public:
        RelationalMessagePassingNeuralNetworkImpl();

        RelationalMessagePassingNeuralNetworkImpl(const PredicateArityList& predicates,
                                                  const DerivedPredicateList& derived_predicates,
                                                  const int32_t hidden_size,
                                                  const int32_t num_layers,
                                                  const bool global_readout,
                                                  const double maximum_smoothness);

        // std::vector<std::pair<std::string, int32_t>> predicates() const
        // {
        //     // TODO: Remove predicate_arities and derive on the fly. Move into source file.
        //     return std::vector<std::pair<std::string, int32_t>>(predicate_arities_.begin(), predicate_arities_.end());
        // }

        // DerivedPredicateList derived_predicates() const
        // {
        //     // TODO: Move into source file.
        //     return external_derived_predicates_;
        // }

        int32_t hidden_size() const { return this->hidden_size_; }

        int32_t number_of_layers() const { return this->num_layers_; }

        bool global_readout() const { return this->global_readout_; }

        double maximum_smoothness() const { return this->maximum_smoothness_; }

      protected:
        torch::Tensor readout_dead_end(const torch::Tensor& object_embeddings, const std::vector<int64_t>& batch_sizes) override;

        torch::Tensor readout_value(const torch::Tensor& object_embeddings, const std::vector<int64_t>& batch_sizes) override;

        std::vector<torch::Tensor> readout_dead_end(const torch::Tensor& object_embeddings,
                                                    const std::vector<int64_t>& batch_sizes,
                                                    const std::vector<std::tuple<int32_t, int32_t, int32_t>>& batch_slices) override;

        std::vector<torch::Tensor> readout_value(const torch::Tensor& object_embeddings,
                                                 const std::vector<int64_t>& batch_sizes,
                                                 const std::vector<std::tuple<int32_t, int32_t, int32_t>>& batch_slices) override;

        std::vector<torch::Tensor> readout_policy(const torch::Tensor& object_embeddings,
                                                  const std::vector<int64_t>& batch_sizes,
                                                  const std::vector<std::tuple<int32_t, int32_t, int32_t>>& batch_slices) override;

      private:
        /**
         * @brief Infer embeddings for all objects.
         *
         * @param batch_states A list of states.
         * @param batch_sizes A list of state sizes in the batch.
         * @return A tensor of all object embeddings.
         */
        torch::Tensor internal_forward(const std::map<std::string, std::vector<int64_t>>& batch_states, const std::vector<int64_t>& batch_sizes) override;
    };

    TORCH_MODULE(RelationalMessagePassingNeuralNetwork);

    std::ostream& operator<<(std::ostream& os, const RelationalMessagePassingNeuralNetwork& network);
}  // namespace models

#endif  // RELATIONAL_MESSAGE_PASSING_NEURAL_NETWORK_HPP_
