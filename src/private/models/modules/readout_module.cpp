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


#include "readout_module.hpp"
#include "torch/torch.h"

ReadoutModuleImpl::ReadoutModuleImpl(const int32_t hidden_size, const int32_t out_size) : value_module_(nullptr), dummy_(torch::empty(0))
{
    value_module_ = register_module("value_module_", MLPModule(hidden_size, out_size));
    dummy_ = register_parameter("dummy_", dummy_, false);
}

torch::Tensor ReadoutModuleImpl::forward(const torch::Tensor& object_embeddings, const std::vector<int64_t>& batch_sizes)
{
    auto cumsum_indices = torch::tensor(batch_sizes).to(this->device(), torch::kInt, true, false).cumsum(0) - 1;  // TODO: This can be computed once.
    auto cumsum_states = object_embeddings.cumsum(0).index_select(0, cumsum_indices);
    auto aggregated_states = torch::cat({ cumsum_states[0].view({ 1, -1 }), cumsum_states.slice(0, 1) - cumsum_states.slice(0, 0, -1) });
    auto output = this->value_module_->forward(aggregated_states);
    return output;

    // Reference implementation below, slower since it involved an explicit loop.

    // const auto object_embeddings_by_state = object_embeddings.split_with_sizes(batch_sizes);
    // std::vector<torch::Tensor> aggregations;
    // for (const auto& state_object_embeddings : object_embeddings_by_state)
    // {
    //     aggregations.push_back(state_object_embeddings.sum(0));
    // }
    // return this->value_module_->forward(torch::stack(aggregations));
}
