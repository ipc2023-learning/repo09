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
#include "mlp_module.hpp"
#include "torch/torch.h"

MLPModuleImpl::MLPModuleImpl() : first_(nullptr), second_(nullptr), reshape_(nullptr) {}

MLPModuleImpl::MLPModuleImpl(int32_t in_features, int32_t out_features) : first_(nullptr), second_(nullptr), reshape_(nullptr)
{
    first_ = register_module("first_", torch::nn::Linear(in_features, in_features));
    second_ = register_module("second_", torch::nn::Linear(in_features, in_features));

    if (in_features != out_features)
    {
        reshape_ = register_module("reshape_", torch::nn::Linear(in_features, out_features));
    }
}

torch::Tensor MLPModuleImpl::forward(const torch::Tensor& input)
{
    const auto output = input + second_->forward(torch::nn::functional::mish(first_->forward(input)));
    return reshape_ ? reshape_->forward(output) : output;
}
