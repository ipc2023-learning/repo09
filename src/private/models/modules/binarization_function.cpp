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
#include "torch/torch.h"

// torch::Tensor BinarizationFunction::forward(torch::autograd::AutogradContext* ctx, torch::Tensor input) { return input; }
// torch::Tensor BinarizationFunction::forward(torch::autograd::AutogradContext* ctx, torch::Tensor input) { return torch::clamp(input, -1.0, 1.0); }  // Works
torch::Tensor BinarizationFunction::forward(torch::autograd::AutogradContext* ctx, torch::Tensor input) { return torch::round(input); }  // Works

torch::autograd::tensor_list BinarizationFunction::backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs)
{
    return { grad_outputs };
}
