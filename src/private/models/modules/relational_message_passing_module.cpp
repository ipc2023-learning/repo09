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


#include "binarization_function.hpp"
#include "relational_message_passing_module.hpp"

RelationalMessagePassingModuleImpl::RelationalMessagePassingModuleImpl(const std::vector<std::pair<int32_t, int32_t>>& id_arities,
                                                                       const int32_t hidden_size,
                                                                       const double maximum_smoothness) :
    hidden_size_(hidden_size),
    relation_modules_(nullptr),
    update_module_(nullptr),
    maximum_smoothness_(maximum_smoothness),
    dummy_(torch::empty(0))
{
    // One module per relation, the input and output size depends on the arity.

    relation_modules_ = register_module("relation_modules_", torch::nn::ModuleList());

    for (auto id_arity : id_arities)
    {
        assert(id_arity.first == (int32_t) this->relation_modules_->size());
        assert(id_arity.second > 0);
        this->relation_modules_->push_back(RelationModule(id_arity.second, hidden_size));
    }

    update_module_ = register_module("update_module_", MLPModule(2 * hidden_size, hidden_size));
    dummy_ = register_parameter("dummy_", dummy_, false);
}

torch::Tensor RelationalMessagePassingModuleImpl::forward(const torch::Tensor& object_embeddings, const std::map<int32_t, torch::Tensor>& relations)
{
    std::vector<torch::Tensor> output_tensors_list;
    std::vector<torch::Tensor> output_indices_list;

    for (uint32_t relation_id = 0; relation_id < this->relation_modules_->size(); ++relation_id)
    {
        const auto relation_module = this->relation_modules_[relation_id];

        if (relation_module != nullptr)
        {
            auto relation_handler = relations.find(relation_id);

            if (relation_handler != relations.end())
            {
                const auto relation_values = relation_handler->second;
                const auto output = relation_module->as<RelationModule>()->forward(object_embeddings, relation_values);
                const auto node_indices = relation_values.view({ -1, 1 }).expand({ -1, this->hidden_size_ });
                output_tensors_list.push_back(output);
                output_indices_list.push_back(node_indices);
            }
        }
    }

    const auto output_tensors = torch::cat(output_tensors_list, 0);
    const auto output_indices = torch::cat(output_indices_list, 0);

    auto exps_max = torch::zeros_like(object_embeddings, torch::TensorOptions(this->device()));
    exps_max.scatter_reduce_(0, output_indices, output_tensors, "amax", false);
    exps_max = exps_max.detach();

    auto exps_sum = torch::full_like(object_embeddings, 1E-16, torch::TensorOptions(this->device()));
    const auto max_offsets = exps_max.gather(0, output_indices).detach();
    const auto exps = (maximum_smoothness_ * (output_tensors - max_offsets)).exp();
    exps_sum.scatter_add_(0, output_indices, exps);

    const auto max_msg = ((1.0 / maximum_smoothness_) * exps_sum.log()) + exps_max;
    const auto next_object_embeddings = this->update_module_->forward(torch::cat({ max_msg, object_embeddings }, 1));
    return next_object_embeddings;
}
