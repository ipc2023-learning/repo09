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


#include "relation_module.hpp"
#include "torch/torch.h"

RelationModuleImpl::RelationModuleImpl(const int32_t arity, const int32_t hidden_size) : relation_(nullptr), arity_(arity), hidden_size_(hidden_size)
{
    auto in_features = arity * hidden_size;
    auto out_features = arity * hidden_size;
    relation_ = register_module("relation_", MLPModule(in_features, out_features));
}

torch::Tensor RelationModuleImpl::forward(const torch::Tensor& object_embeddings, const torch::Tensor& relation_values)
{
    auto in_features = this->arity_ * this->hidden_size_;
    auto input = object_embeddings.index_select(0, relation_values).view({ -1, in_features });
    auto output = this->relation_->forward(input).view({ -1, this->hidden_size_ });
    return output;
}
