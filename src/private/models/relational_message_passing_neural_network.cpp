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
#include "modules/binarization_function.hpp"
#include "modules/readout_module.hpp"
#include "modules/relational_message_passing_module.hpp"
#include "relational_message_passing_neural_network.hpp"
#include "torch/torch.h"

#include <numeric>
#include <string>

void print_tensor(const torch::Tensor& tensor)
{
    std::cout << "---" << std::endl;
    std::cout << tensor.sizes() << std::endl;
    torch::print(tensor);
    std::cout << std::endl;
}

void print_batch(const std::map<std::string, std::vector<int64_t>>& batch_states)
{
    std::cout << "Batch: (" << batch_states.size() << "):" << std::endl;
    for (const auto& entry : batch_states)
    {
        const auto& predicate_name = entry.first;
        const auto& ids = entry.second;
        std::cout << " - " << predicate_name << ": " << ids.size() << " [";

        for (const auto& id : ids)
        {
            std::cout << " " << id;
        }

        std::cout << " ]" << std::endl;
    }
    std::cout << std::endl;
}

void print_batch(const std::map<std::string, std::vector<int64_t>>& batch_states, const std::map<std::string, int32_t>& object_ids)
{
    std::map<int32_t, std::string> id_to_name;

    for (const auto& entry : object_ids)
    {
        id_to_name.insert(std::make_pair(entry.second, entry.first));
    }

    std::cout << "Batch: (" << batch_states.size() << "):" << std::endl;

    for (const auto& entry : batch_states)
    {
        const auto& predicate_name = entry.first;
        const auto& ids = entry.second;
        std::cout << " - " << predicate_name << ": " << ids.size() << " [";

        for (const auto& id : ids)
        {
            std::cout << " " << id_to_name.at(id);
        }

        std::cout << " ]" << std::endl;
    }
    std::cout << std::endl;
}

void print_grouped_atoms(const std::map<formalism::Predicate, formalism::AtomList>& grouped_atoms)
{
    std::cout << std::endl;
    for (const auto& entry : grouped_atoms)
    {
        const auto& predicate = entry.first;
        const auto& atoms = entry.second;

        std::cout << predicate << ": " << atoms << std::endl;
    }
    std::cout << std::endl;
}

namespace models
{
    RelationalMessagePassingNeuralNetworkImpl::RelationalMessagePassingNeuralNetworkImpl() :
        RelationalNeuralNetworkBase(),
        hidden_size_(0),
        num_layers_(0),
        global_readout_(false),
        maximum_smoothness_(12.0),
        message_passing_module_(RelationalMessagePassingModule()),
        global_module_(),
        transition_module_(),
        readout_global_module_(),
        readout_transition_module_(),
        readout_value_module_(),
        readout_dead_end_module_()
    {
    }

    RelationalMessagePassingNeuralNetworkImpl::RelationalMessagePassingNeuralNetworkImpl(const PredicateArityList& predicates,
                                                                                         const DerivedPredicateList& derived_predicates,
                                                                                         const int32_t hidden_size,
                                                                                         const int32_t num_layers,
                                                                                         const bool global_readout,
                                                                                         const double maximum_smoothness) :
        RelationalNeuralNetworkBase(predicates, derived_predicates),
        hidden_size_(hidden_size),
        num_layers_(num_layers),
        global_readout_(global_readout),
        maximum_smoothness_(maximum_smoothness),
        message_passing_module_(nullptr),
        global_module_(nullptr),
        transition_module_(nullptr),
        readout_global_module_(nullptr),
        readout_transition_module_(nullptr),
        readout_value_module_(nullptr),
        readout_dead_end_module_(nullptr)
    {
        message_passing_module_ = register_module("message_passing_module_", RelationalMessagePassingModule(id_arities(), hidden_size, maximum_smoothness));
        global_module_ = register_module("global_module_", MLPModule(2 * hidden_size, hidden_size));
        transition_module_ = register_module("transition_module_", MLPModule(2 * hidden_size, 2 * hidden_size));
        readout_global_module_ = register_module("readout_layer_module_", ReadoutModule(hidden_size, hidden_size));
        readout_transition_module_ = register_module("readout_transition_module_", ReadoutModule(2 * hidden_size, 1));
        readout_value_module_ = register_module("readout_value_module_", ReadoutModule(hidden_size, 1));
        readout_dead_end_module_ = register_module("readout_dead_end_module_", ReadoutModule(hidden_size, 1));
    }

    torch::Tensor RelationalMessagePassingNeuralNetworkImpl::readout_dead_end(const torch::Tensor& object_embeddings, const std::vector<int64_t>& batch_sizes)
    {
        return readout_dead_end_module_->forward(object_embeddings, batch_sizes);
    }

    torch::Tensor RelationalMessagePassingNeuralNetworkImpl::readout_value(const torch::Tensor& object_embeddings, const std::vector<int64_t>& batch_sizes)
    {
        return readout_value_module_->forward(object_embeddings, batch_sizes);
    }

    std::vector<torch::Tensor>
    RelationalMessagePassingNeuralNetworkImpl::readout_dead_end(const torch::Tensor& object_embeddings,
                                                                const std::vector<int64_t>& batch_sizes,
                                                                const std::vector<std::tuple<int32_t, int32_t, int32_t>>& batch_slices)
    {
        std::vector<torch::Tensor> dead_end_vector;
        const auto dead_ends = readout_dead_end_module_->forward(object_embeddings, batch_sizes);

        for (const auto& slice : batch_slices)
        {
            const auto slice_start = std::get<0>(slice);
            const auto slice_end = std::get<1>(slice);

            dead_end_vector.push_back(dead_ends.index({ torch::indexing::Slice(slice_start, slice_end) }));
        }

        return dead_end_vector;
    }

    std::vector<torch::Tensor> RelationalMessagePassingNeuralNetworkImpl::readout_value(const torch::Tensor& object_embeddings,
                                                                                        const std::vector<int64_t>& batch_sizes,
                                                                                        const std::vector<std::tuple<int32_t, int32_t, int32_t>>& batch_slices)
    {
        std::vector<torch::Tensor> value_vector;
        const auto values = readout_value_module_->forward(object_embeddings, batch_sizes);

        for (const auto& slice : batch_slices)
        {
            const auto slice_start = std::get<0>(slice);
            const auto slice_end = std::get<1>(slice);

            value_vector.push_back(values.index({ torch::indexing::Slice(slice_start, slice_end) }));
        }

        return value_vector;
    }

    std::vector<torch::Tensor> RelationalMessagePassingNeuralNetworkImpl::readout_policy(const torch::Tensor& object_embeddings,
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
                const auto policy_scores = readout_transition_module_->forward(transition_module_->forward(policy_object_embeddings), slice_sizes);
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

    torch::Tensor RelationalMessagePassingNeuralNetworkImpl::internal_forward(const std::map<std::string, std::vector<int64_t>>& batch_states,
                                                                              const std::vector<int64_t>& batch_sizes)
    {
        // Translate input to a more convenient format for internal computations.

        std::map<int32_t, torch::Tensor> internal_relations;
        const auto pids = predicate_ids();

        for (const auto& name_values : batch_states)
        {
            const auto id_handler = pids.find(name_values.first);

            if (id_handler != pids.end())
            {
                const auto id = id_handler->second;
                const auto relation_values = torch::tensor(name_values.second).to(this->device(), torch::kInt64, true, false);
                internal_relations.insert(std::make_pair(id, relation_values));
            }
        }

        // Initialize object embeddings

        const auto num_states = std::accumulate(batch_sizes.begin(), batch_sizes.end(), 0);
        auto object_embeddings = torch::zeros({ num_states, this->hidden_size_ }).to(this->device(), torch::kFloat, true, false);

        // Random node initialization grants additional expressive power to states of size similar to the training set.
        // However, we are interested in generalization to arbitary sizes, but this is how one would implemented it:

        // const auto init_zeroes =
        //     torch::zeros({ num_states, (this->hidden_size_ / 2) + (this->hidden_size_ % 2) }).to(this->device(), torch::kFloat, true, false);
        // const auto init_random = torch::randn({ num_states, (this->hidden_size_ / 2) }).to(this->device(), torch::kFloat, true, false);
        // auto object_embeddings = torch::cat({ init_zeroes, init_random }, 1);

        // Run message passing and then do global readouts

        for (int32_t iteration = 0; iteration < num_layers_; ++iteration)
        {
            object_embeddings = this->message_passing_module_->forward(object_embeddings, internal_relations);

            if (this->global_readout_)
            {
                throw std::runtime_error("not implemented");
                // const auto readout_embeddings = this->readout_global_module_->forward(object_embeddings, batch_sizes);
                //... repeat_interleave
                // object_embeddings = this->global_module_(/*...*/);
            }

            // object_embeddings = BinarizationFunction().apply(object_embeddings);
            // object_embeddings = torch::round(object_embeddings);  // Does NOT work!

            // const auto original_size = object_embeddings.sizes();
            // const auto logits = object_embeddings.view({ -1, 1 });
            // const auto zeros = torch::zeros_like(logits);
            // const auto logits_with_zeros = torch::cat({ logits, zeros }, 1);
            // const auto binarization_with_zeros =
            //     torch::nn::functional::gumbel_softmax(logits_with_zeros, torch::nn::functional::GumbelSoftmaxFuncOptions().hard(true).dim(1));
            // const auto binarization = binarization_with_zeros.t().split(1)[0];
            // object_embeddings = binarization.t().view(original_size);
        }

        return object_embeddings;
    }

    std::ostream& operator<<(std::ostream& os, const RelationalMessagePassingNeuralNetwork& network)
    {
        std::cout << "Relational Message Passing Neural Network:" << std::endl;
        std::cout << " - # features: " << network->hidden_size() << std::endl;
        std::cout << " - # layers: " << network->number_of_layers() << std::endl;
        std::cout << " - Global readout: " << (network->global_readout() ? "Yes" : "No") << std::endl;
        std::cout << " - Maximum smoothness: " << network->maximum_smoothness() << std::endl;
        std::cout << " - Predicates: ";
        const auto predicates = network->predicates();
        for (uint32_t index = 0; index < predicates.size(); ++index)
        {
            const auto& predicate = predicates[index];

            std::cout << predicate.first + "/" + std::to_string(predicate.second);

            if (index + 1 < predicates.size())
            {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;

        return os;
    }
}  // namespace models
