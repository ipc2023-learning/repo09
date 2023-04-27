/*
 * Copyright (C) 2023 Simon Stahlberg
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */


#include "../formalism/atom.hpp"
#include "../formalism/problem.hpp"
#include "../formalism/state.hpp"
#include "relational_transformer.hpp"
#include "torch/torch.h"

#include <numeric>
#include <string>

namespace models
{
    RelationalTransformerImpl::RelationalTransformerImpl() :
        RelationalNeuralNetworkBase(),
        num_layers_(0),
        embedding_size_(0),
        num_identifiers_(0),
        transition_module_(nullptr),
        readout_transition_module_(nullptr),
        readout_value_module_(nullptr),
        readout_dead_end_module_(nullptr),
        type_layer_(nullptr),
        label_layer_(nullptr),
        identifier_layer_(nullptr),
        type_embeddings_(nullptr),
        label_predicate_embeddings_(nullptr),
        label_index_embeddings_(nullptr),
        label_null_embedding_(nullptr),
        token_graph_embedding_(nullptr),
        token_null_embedding_(nullptr),
        // identifier_embeddings_(nullptr),
        transformer_encoder_(nullptr)

    {
    }

    RelationalTransformerImpl::RelationalTransformerImpl(const PredicateArityList& predicates,
                                                         const DerivedPredicateList& derived_predicates,
                                                         const int64_t num_layers,
                                                         const int64_t num_identifiers,
                                                         const int64_t min_embedding_size,
                                                         const int64_t min_attention_heads) :
        RelationalNeuralNetworkBase(predicates, derived_predicates),
        num_layers_(num_layers),
        num_attention_heads_(0),
        embedding_size_(0),
        num_identifiers_(num_identifiers),
        transition_module_(nullptr),
        readout_transition_module_(nullptr),
        readout_value_module_(nullptr),
        readout_dead_end_module_(nullptr),
        type_layer_(nullptr),
        label_layer_(nullptr),
        identifier_layer_(nullptr),
        type_embeddings_(nullptr),
        label_predicate_embeddings_(nullptr),
        label_index_embeddings_(nullptr),
        label_null_embedding_(nullptr),
        token_graph_embedding_(nullptr),
        token_null_embedding_(nullptr),
        // identifier_embeddings_(nullptr),
        transformer_encoder_(nullptr)
    {
        if (num_identifiers < 1)
        {
            throw std::invalid_argument("\"num_identifiers\" must be positive");
        }

        if (min_embedding_size < 1)
        {
            throw std::invalid_argument("\"min_embedding_size\" must be positive");
        }

        if (min_attention_heads < 1)
        {
            throw std::invalid_argument("\"min_attention_heads\" must be positive");
        }

        const auto num_attention_heads = min_attention_heads + ((min_attention_heads % 2) ? 1 : 0);
        const auto embedding_size = ((min_embedding_size + num_attention_heads - 1) / num_attention_heads) * num_attention_heads;

        num_attention_heads_ = num_attention_heads;
        embedding_size_ = embedding_size;

        int64_t max_arity = 0;
        for (const auto& [_, arity] : predicates)
        {
            max_arity = std::max(max_arity, static_cast<int64_t>(arity));
        }

        transition_module_ = register_module("transition_module_", MLPModule(2 * embedding_size, 2 * embedding_size));
        readout_transition_module_ = register_module("readout_transition_module_", MLPModule(embedding_size, 1));
        readout_value_module_ = register_module("readout_value_module_", MLPModule(embedding_size, 1));
        readout_dead_end_module_ = register_module("readout_dead_end_module_", MLPModule(embedding_size, 1));

        type_layer_ = register_module("type_layer_", torch::nn::Linear(embedding_size, embedding_size));
        label_layer_ = register_module("label_layer_", torch::nn::Linear(embedding_size, embedding_size));
        identifier_layer_ = register_module("identifier_layer_", torch::nn::Linear(embedding_size, embedding_size));

        const int64_t num_types = 3;
        const auto num_predicates = 2 * predicates.size();  // Predicates does not include goal predicates.

        type_embeddings_ = register_module("type_embeddings_", torch::nn::Embedding(num_types, embedding_size));
        label_predicate_embeddings_ = register_module("label_predicate_embeddings_", torch::nn::Embedding(num_predicates, embedding_size));
        label_index_embeddings_ = register_module("label_index_embeddings_", torch::nn::Embedding(max_arity, embedding_size));
        label_null_embedding_ = register_module("label_null_embedding_", torch::nn::Embedding(1, embedding_size));
        token_graph_embedding_ = register_module("token_graph_embedding_", torch::nn::Embedding(1, embedding_size));
        token_null_embedding_ = register_module("token_null_embedding_", torch::nn::Embedding(1, embedding_size));
        // identifier_embeddings_ = register_module("identifier_embeddings_", torch::nn::Embedding(num_identifiers_, embedding_size / 2));

        torch::nn::init::xavier_uniform_(type_embeddings_->weight);
        torch::nn::init::xavier_uniform_(label_predicate_embeddings_->weight);
        torch::nn::init::xavier_uniform_(label_index_embeddings_->weight);
        torch::nn::init::xavier_uniform_(label_null_embedding_->weight);
        torch::nn::init::xavier_uniform_(token_graph_embedding_->weight);
        torch::nn::init::xavier_uniform_(token_null_embedding_->weight);
        // torch::nn::init::xavier_uniform_(identifier_embeddings_->weight);

        const auto layer_options =
            torch::nn::TransformerEncoderLayerOptions(embedding_size, num_attention_heads).dim_feedforward(embedding_size).activation(torch::nn::Mish());
        const auto encoder_options = torch::nn::TransformerEncoderOptions(layer_options, num_layers);
        transformer_encoder_ = register_module("transformer_encoder_", torch::nn::TransformerEncoder(encoder_options));
    }

    torch::Tensor RelationalTransformerImpl::readout_dead_end(const torch::Tensor& object_embeddings, const std::vector<int64_t>& batch_sizes)
    {
        return readout_dead_end_module_->forward(object_embeddings);
    }

    torch::Tensor RelationalTransformerImpl::readout_value(const torch::Tensor& object_embeddings, const std::vector<int64_t>& batch_sizes)
    {
        return readout_value_module_->forward(object_embeddings);
    }

    std::vector<torch::Tensor> RelationalTransformerImpl::readout_dead_end(const torch::Tensor& object_embeddings,
                                                                           const std::vector<int64_t>& batch_sizes,
                                                                           const std::vector<std::tuple<int32_t, int32_t, int32_t>>& batch_slices)
    {
        std::vector<torch::Tensor> dead_end_vector;
        const auto dead_ends = readout_dead_end_module_->forward(object_embeddings);

        for (const auto& slice : batch_slices)
        {
            const auto slice_start = std::get<0>(slice);
            const auto slice_end = std::get<1>(slice);

            dead_end_vector.push_back(dead_ends.index({ torch::indexing::Slice(slice_start, slice_end) }));
        }

        return dead_end_vector;
    }

    std::vector<torch::Tensor> RelationalTransformerImpl::readout_value(const torch::Tensor& object_embeddings,
                                                                        const std::vector<int64_t>& batch_sizes,
                                                                        const std::vector<std::tuple<int32_t, int32_t, int32_t>>& batch_slices)
    {
        std::vector<torch::Tensor> value_vector;
        const auto values = readout_value_module_->forward(object_embeddings);

        for (const auto& slice : batch_slices)
        {
            const auto slice_start = std::get<0>(slice);
            const auto slice_end = std::get<1>(slice);

            value_vector.push_back(values.index({ torch::indexing::Slice(slice_start, slice_end) }));
        }

        return value_vector;
    }

    std::vector<torch::Tensor> RelationalTransformerImpl::readout_policy(const torch::Tensor& object_embeddings,
                                                                         const std::vector<int64_t>& batch_sizes,
                                                                         const std::vector<std::tuple<int32_t, int32_t, int32_t>>& batch_slices)
    {
        std::vector<torch::Tensor> policy_vector;

        int32_t object_offset = 0;
        for (const auto& slice : batch_slices)
        {
            const auto slice_start = std::get<0>(slice);
            const auto slice_end = std::get<1>(slice);
            const auto slice_objects = std::get<2>(slice);
            const auto slice_size = slice_end - slice_start;
            const auto num_successors = slice_size - 1;
            const auto slice_objects_start = object_offset;
            const auto slice_objects_end = slice_objects_start + slice_size * slice_objects;
            object_offset = slice_objects_end;

            const auto state = object_embeddings.index({ torch::indexing::Slice(slice_objects_start, slice_objects_start + slice_objects) });
            const auto successors = object_embeddings.index({ torch::indexing::Slice(slice_objects_start + slice_objects, slice_objects_end) });

            if (num_successors > 0)
            {
                std::vector<int64_t> slice_sizes;
                for (int32_t successor_index = 0; successor_index < num_successors; ++successor_index)
                {
                    slice_sizes.push_back(slice_objects);
                }

                const auto policy_object_embeddings = torch::cat({ state.repeat({ num_successors, 1 }), successors }, 1);
                const auto policy_scores = readout_transition_module_->forward(transition_module_->forward(policy_object_embeddings));  // TODO: Correct?
                // const auto policy_scores = readout_transition_module_->forward(transition_module_->forward(policy_object_embeddings), slice_sizes);
                const auto policy_distribution = policy_scores.softmax(0);
                policy_vector.push_back(policy_distribution);
            }
            else
            {
                policy_vector.push_back(torch::empty(0, this->device()));
            }
        }

        return policy_vector;
    }

    torch::Tensor RelationalTransformerImpl::pad_and_stack_tensors(const std::vector<torch::Tensor>& tensors, int64_t max_length)
    {
        std::vector<torch::Tensor> padded_tensors;

        for (const auto& tensor : tensors)
        {
            const auto pad_length = max_length - tensor.size(0);
            const auto embedding_size = tensor.size(1);
            const auto options = torch::dtype(tensor.dtype()).device(tensor.device());
            const auto padding = torch::zeros({ pad_length, embedding_size }, options);
            padded_tensors.push_back(torch::cat({ tensor, padding }));
        }

        return torch::stack(padded_tensors);
    }

    torch::Tensor RelationalTransformerImpl::create_attention_masks(const std::vector<torch::Tensor>& tensors, int64_t max_length)
    {
        std::vector<torch::Tensor> masks;

        for (const auto& tensor : tensors)
        {
            const auto pad_length = max_length - tensor.size(0);
            const auto options = torch::dtype(torch::kBool).device(tensor.device());
            auto mask = torch::cat({ torch::zeros({ tensor.size(0) }, options), torch::ones({ pad_length }, options) });
            masks.push_back(mask);
        }

        return torch::stack(masks);
    }

    torch::Tensor RelationalTransformerImpl::internal_forward(const std::map<std::string, std::vector<int64_t>>& batch_states,
                                                              const std::vector<int64_t>& batch_sizes)
    {
        std::vector<torch::Tensor> batch_identifier_embeddings;
        std::vector<torch::Tensor> batch_label_embeddings;
        std::vector<torch::Tensor> batch_type_embeddings;

        std::vector<std::map<std::string, std::vector<int64_t>>> batch_split(batch_sizes.size());

        {  // Transform batch_states and batch_sizes into a more convenient format.

            // The relations in batch_states are derived from multiple states by introducing new object symbols that are distinct from those in the original
            // states. For example, consider the case where the values for a binary relation r are [1, 2, 3, 4, 5, 6], and batch_sizes is [2, 2, 2]. In this
            // case, the atom r(1, 2) appears in all three states. The following code reorganizes batch_states by splitting it into batch_size different
            // vectors.

            std::vector<int64_t> batch_ranges(batch_sizes.size() + 1, 0);
            std::partial_sum(batch_sizes.begin(), batch_sizes.end(), batch_ranges.begin() + 1);

            const auto num_object_ids = batch_ranges[batch_ranges.size() - 1];
            std::vector<std::size_t> batch_indices(num_object_ids);

            for (std::size_t batch_index = 0; batch_index < batch_sizes.size(); ++batch_index)
            {
                std::fill(batch_indices.begin() + batch_ranges[batch_index], batch_indices.begin() + batch_ranges[batch_index + 1], batch_index);
            }

            for (const auto& [predicate, values] : batch_states)
            {
                for (auto& split_values : batch_split)
                {
                    split_values.insert(std::make_pair(predicate, std::vector<int64_t>()));
                }

                for (const auto& value : values)
                {
                    const auto batch_index = batch_indices[value];
                    batch_split[batch_index][predicate].push_back(value - batch_ranges[batch_index]);
                }
            }
        }

        auto pred_arities = predicate_arities();
        auto& pred_ids = predicate_ids();

        for (const auto& [predicate, arity] : predicates())
        {
            pred_arities.insert(std::make_pair(predicate + "_goal", arity));
        }

        const auto zero_index = torch::tensor(0, { device() });
        const auto type_object = type_embeddings_->forward(zero_index + 0);
        const auto type_atom = type_embeddings_->forward(zero_index + 1);
        const auto type_term = type_embeddings_->forward(zero_index + 2);
        const auto label_null = label_null_embedding_->forward(zero_index);

        for (std::size_t batch_index = 0; batch_index < batch_sizes.size(); ++batch_index)
        {
            // Suppose we have a set of atoms A over a set of objects O. We can construct a graph G=(V, E) where V is the union of O and A, and an edge in E is
            // always between an object vertex and an atom vertex. Each vertex has a label that indicates whether it is an object or an atom. If it is an atom,
            // the label also includes the predicate name. Edges are also labeled with the position of the object mentioned in the atom, given by an index in
            // the range [0, the arity of the atom). It's important to note that the set of atoms, batch_states, include goal atoms as well.

            const auto num_objects = batch_sizes[batch_index];
            const auto& predicate_atoms = batch_split[batch_index];

            // Each vertex in the graph also has a unique identifier, represented by a unit vector that is orthogonal to all other identifiers. To generate
            // these identifiers, we perform QR decomposition on a random matrix and use the resulting Q matrix to sample the identifiers. Note that each
            // identifier has a fixed size, and if we need more identifiers than the fixed size allows, we trim them, which may lead to two identifiers that are
            // not perfectly orthogonal. Nonetheless, we expect this strategy to produce identifiers that are close to orthogonal, meaning that the dot product
            // is close to zero.

            int64_t num_atoms = 0;

            for (const auto& [predicate, values] : predicate_atoms)
            {
                const auto arity = pred_arities.at(predicate);

                if (arity > 0)
                {
                    num_atoms += values.size() / arity;
                }
            }

            const auto num_vertices = num_objects + num_atoms;
            const auto padding = std::max((embedding_size_ / 2) - num_vertices, static_cast<int64_t>(0));
            const auto random_matrix = torch::randn({ num_vertices, num_vertices }, { device() });
            const auto q = std::get<0>(torch::linalg::qr(random_matrix, "reduced")).t();
            const auto q_slices = q.slice(0, 0, num_vertices);
            const auto q_resized = torch::constant_pad_nd(q_slices, { 0, padding }).narrow(1, 0, embedding_size_ / 2);
            const auto vertex_identifiers = (q_resized / (torch::linalg::vector_norm(q_resized, 2, 1, true, {}) + 1E-16));  // Normalize vectors.

            // const auto permuted_indices = torch::randperm(num_identifiers_, torch::dtype(torch::kInt64).device(device()));
            // const auto vertex_identifiers = identifier_embeddings_->forward(permuted_indices);

            std::vector<torch::Tensor> identifier_embeddings;
            std::vector<torch::Tensor> label_embeddings;
            std::vector<torch::Tensor> type_embeddings;

            // Generate all embeddings for vertices and edges.
            // The identifiers in the range [0, num_object) are reserved for objects, and the remaining identifiers are used for atoms.

            for (int64_t object_index = 0; object_index < num_objects; ++object_index)
            {
                const auto obj_identifier = vertex_identifiers[object_index];
                identifier_embeddings.push_back(torch::cat({ obj_identifier, obj_identifier }, 0));
                label_embeddings.push_back(label_null);
                type_embeddings.push_back(type_object);
            }

            int64_t atom_index_offset = num_objects;
            for (const auto& [predicate, values] : predicate_atoms)
            {
                const auto arity = (int64_t) pred_arities.at(predicate);

                if (arity > 0)
                {
                    const auto num_atoms = ((int64_t) values.size()) / arity;

                    for (int64_t atom_index = 0; atom_index < num_atoms; ++atom_index)
                    {
                        const auto label_predicate = label_predicate_embeddings_->forward(zero_index + pred_ids.at(predicate));
                        const auto atom_identifier = vertex_identifiers[atom_index_offset + atom_index];
                        identifier_embeddings.push_back(torch::cat({ atom_identifier, atom_identifier }, 0));
                        label_embeddings.push_back(label_predicate);
                        type_embeddings.push_back(type_atom);

                        for (int64_t edge_index = 0; edge_index < arity; ++edge_index)
                        {
                            const auto value_index = atom_index * arity + edge_index;
                            const auto object_identifier = vertex_identifiers[values[value_index]];
                            const auto label_index = label_index_embeddings_->forward(zero_index + edge_index);
                            identifier_embeddings.push_back(torch::cat({ object_identifier, atom_identifier }, 0));
                            label_embeddings.push_back(label_index);
                            type_embeddings.push_back(type_term);

                            // TODO: It's unclear if we need two tokens per undirected edge...
                            identifier_embeddings.push_back(torch::cat({ atom_identifier, object_identifier }, 0));
                            label_embeddings.push_back(label_index);
                            type_embeddings.push_back(type_term);
                        }
                    }

                    atom_index_offset += num_atoms;
                }
                else
                {
                    // TODO: Implement support for nullary atoms.
                }
            }

            batch_identifier_embeddings.push_back(torch::stack(identifier_embeddings));
            batch_label_embeddings.push_back(torch::stack(label_embeddings));
            batch_type_embeddings.push_back(torch::stack(type_embeddings));
        }

        // In the C++ version, the input to MHA must have the shape (seq, batch, features).
        // The result is then converted back to (batch, seq, features) and the object embeddings are extracted.

        int64_t max_length = 0;
        for (const auto& tensor : batch_identifier_embeddings)
        {
            max_length = std::max(max_length, tensor.size(0));
        }

        const auto input_identifiers = identifier_layer_->forward(pad_and_stack_tensors(batch_identifier_embeddings, max_length));
        const auto input_labels = label_layer_->forward(pad_and_stack_tensors(batch_label_embeddings, max_length));
        const auto input_types = type_layer_->forward(pad_and_stack_tensors(batch_type_embeddings, max_length));

        auto input = input_identifiers + input_labels + input_types;
        auto input_masks = create_attention_masks(batch_identifier_embeddings, max_length);

        // // ----------------------------------------
        // // Aggregate the embeddings for all objects
        // // ----------------------------------------

        // const auto output = transformer_encoder_->forward(input.permute({ 1, 0, 2 }), {}, input_masks).permute({ 1, 0, 2 });

        // std::vector<torch::Tensor> batch_output;

        // for (std::size_t batch_index = 0; batch_index < batch_sizes.size(); ++batch_index)
        // {
        //     const auto num_objects = batch_sizes[batch_index];
        //     batch_output.emplace_back(output[batch_index].slice(0, 0, num_objects).sum(0));
        // }

        // return torch::stack(batch_output);

        // ----------------------------------
        // Use a single graph token as output
        // ----------------------------------

        {  // Add graph and null token
            const auto batch_size = input_labels.size(0);
            const auto batch_zeros = torch::zeros({ batch_size }, torch::dtype(torch::kInt64).device(device()));
            const auto graph_tokens = token_graph_embedding_->forward(batch_zeros).view({ batch_size, 1, -1 });
            const auto null_tokens = token_null_embedding_->forward(batch_zeros).view({ batch_size, 1, -1 });
            const auto token_mask = batch_zeros.view({ batch_size, 1 }) != 0;

            input = torch::cat({ graph_tokens, null_tokens, input }, 1);
            input_masks = torch::cat({ token_mask, token_mask, input_masks }, 1);
        }

        const auto output = transformer_encoder_->forward(input.permute({ 1, 0, 2 }), {}, input_masks).permute({ 1, 0, 2 });
        const auto graph_token_embedding = output.slice(1, 0, 1).squeeze(1);
        return graph_token_embedding;
    }

    std::ostream& operator<<(std::ostream& os, const RelationalTransformer& network)
    {
        std::cout << "Relational Transformer:" << std::endl;
        std::cout << " - Layers: " << network->num_layers() << std::endl;
        std::cout << " - Embedding size: " << network->embedding_size() << std::endl;
        std::cout << " - Attention heads: " << network->num_attention_heads() << std::endl;
        std::cout << " - Predicates: ";
        const auto predicates = network->predicates();
        for (uint32_t index = 0; index < predicates.size(); ++index)
        {
            const auto& [name, arity] = predicates[index];

            std::cout << name + "/" + std::to_string(arity);

            if (index + 1 < predicates.size())
            {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;

        return os;
    }
}  // namespace models
