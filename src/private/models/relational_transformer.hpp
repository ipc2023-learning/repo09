#if !defined(RELATIONAL_TRANSFORMER_HPP_)
#define RELATIONAL_TRANSFORMER_HPP_

#include "../formalism/atom.hpp"
#include "../formalism/declarations.hpp"
#include "../formalism/state.hpp"
#include "modules/multihead_attention_module.hpp"
#include "modules/readout_module.hpp"
#include "relational_neural_network_base.hpp"
#include "torch/torch.h"

#include <string>
#include <vector>

namespace models
{
    class RelationalTransformerImpl : public RelationalNeuralNetworkBase
    {
      private:
        int64_t num_layers_;
        int64_t num_attention_heads_;
        int64_t embedding_size_;
        int64_t num_identifiers_;
        MLPModule transition_module_;
        MLPModule readout_transition_module_;
        MLPModule readout_value_module_;
        MLPModule readout_dead_end_module_;
        torch::nn::Linear type_layer_;
        torch::nn::Linear label_layer_;
        torch::nn::Linear identifier_layer_;
        torch::nn::Embedding type_embeddings_;
        torch::nn::Embedding label_predicate_embeddings_;
        torch::nn::Embedding label_index_embeddings_;
        torch::nn::Embedding label_null_embedding_;
        torch::nn::Embedding token_graph_embedding_;
        torch::nn::Embedding token_null_embedding_;
        // torch::nn::Embedding identifier_embeddings_;
        torch::nn::TransformerEncoder transformer_encoder_;

      public:
        RelationalTransformerImpl();

        RelationalTransformerImpl(const PredicateArityList& predicates,
                                  const DerivedPredicateList& derived_predicates,
                                  const int64_t num_layers,
                                  const int64_t num_identifiers,
                                  const int64_t min_embedding_size,
                                  const int64_t min_attention_heads);

        inline int64_t num_layers() const { return this->num_layers_; }

        inline int64_t num_identifiers() const { return this->num_identifiers_; }

        inline int64_t num_attention_heads() const { return this->num_attention_heads_; }

        inline int64_t embedding_size() const { return this->embedding_size_; }

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
        torch::Tensor pad_and_stack_tensors(const std::vector<torch::Tensor>& tensors, int64_t max_length);

        torch::Tensor create_attention_masks(const std::vector<torch::Tensor>& tensors, int64_t max_length);

        /**
         * @brief Infer embeddings for all objects.
         *
         * @param batch_states A list of states.
         * @param batch_sizes A list of state sizes in the batch.
         * @return A tensor of all object embeddings.
         */
        torch::Tensor internal_forward(const std::map<std::string, std::vector<int64_t>>& batch_states, const std::vector<int64_t>& batch_sizes) override;
    };

    TORCH_MODULE(RelationalTransformer);

    std::ostream& operator<<(std::ostream& os, const RelationalTransformer& network);
}  // namespace models

#endif  // RELATIONAL_TRANSFORMER_HPP_
