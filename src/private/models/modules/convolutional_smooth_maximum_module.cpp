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
#include "convolutional_smooth_maximum_module.hpp"
#include "torch/torch.h"

ConvolutionalSmoothMaximumModuleImpl::ConvolutionalSmoothMaximumModuleImpl() : alpha_(0), convolution_(nullptr) {}

ConvolutionalSmoothMaximumModuleImpl::ConvolutionalSmoothMaximumModuleImpl(int32_t in_features, int32_t out_features, double alpha) :
    alpha_(alpha),
    convolution_(nullptr)
{
    const auto options = torch::nn::Conv2dOptions(in_features, out_features, { 3, 3 }).padding(1).bias(false);
    convolution_ = register_module("convolution_", torch::nn::Conv2d(options));
}

torch::Tensor ConvolutionalSmoothMaximumModuleImpl::forward(const torch::Tensor& input)
{
    const auto detached_input = input.detach();
    const auto max = detached_input.max();
    const auto min = detached_input.min();

    const auto norm_zero_one = (input - min) / ((max + 1E-16) - min);
    const auto interval_size = 10.0;
    const auto norm_up_to_zero = (norm_zero_one - 1.0) * interval_size;
    const auto exps = (alpha_ * norm_up_to_zero).exp();
    const auto smooth_max = ((((convolution_->forward(exps) - alpha_ * -interval_size).log() / alpha_) / interval_size) + 1.0) * (max - min) + min;

    return smooth_max;
}
