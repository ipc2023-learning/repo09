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
#include "convolutional_neural_network.hpp"
#include "torch/torch.h"

#include <numeric>
#include <string>

namespace models
{
    ConvolutionalNeuralNetworkImpl::ConvolutionalNeuralNetworkImpl() :
        initialization_(nullptr),
        message_(nullptr),
        update_(nullptr),
        readout_(nullptr),
        aggregation_(nullptr),
        layers_(0),
        dummy_(torch::empty(0))
    {
    }

    ConvolutionalNeuralNetworkImpl::ConvolutionalNeuralNetworkImpl(uint32_t height, uint32_t width, uint32_t colors, uint32_t layers, uint32_t channels) :
        initialization_(nullptr),
        message_(nullptr),
        update_(nullptr),
        readout_(nullptr),
        aggregation_(nullptr),
        layers_(layers),
        dummy_(torch::empty(0))
    {
        const auto upscale = torch::nn::Conv2dOptions(colors, channels, { 1, 1 });
        const auto conv = torch::nn::Conv2dOptions(channels, channels, { 1, 1 });
        const auto update = torch::nn::Conv2dOptions(2 * channels, 2 * channels, { 1, 1 });
        const auto downscale = torch::nn::Conv2dOptions(2 * channels, channels, { 1, 1 });

        initialization_ = register_module("initialization_", torch::nn::Sequential(torch::nn::Conv2d(upscale), torch::nn::ReLU(), torch::nn::Conv2d(conv)));
        message_ = register_module("message_", torch::nn::Sequential(torch::nn::Conv2d(conv), torch::nn::ReLU(), torch::nn::Conv2d(conv)));
        update_ = register_module("update_", torch::nn::Sequential(torch::nn::Conv2d(update), torch::nn::ReLU(), torch::nn::Conv2d(downscale)));

        // This readout function tend to not generalize well since the location of each pixel is important.
        // readout_ = register_module("readout_",
        //                            torch::nn::Sequential(torch::nn::Flatten(),
        //                                                  torch::nn::Linear(height * width * channels, channels),
        //                                                  torch::nn::ReLU(),
        //                                                  torch::nn::Linear(channels, 1)));

        // This readout function has better generalization since it is permutation invariant, i.e., doesn't care about the location of each pixel.
        readout_ = register_module("readout_", torch::nn::Sequential(torch::nn::Linear(channels, channels), torch::nn::ReLU(), torch::nn::Linear(channels, 1)));

        aggregation_ = register_module("aggregation_", ConvolutionalSmoothMaximumModule(channels, channels, 64));
        // const auto aggregate = torch::nn::Conv2dOptions(channels, channels, { 3, 3 }).padding(1).bias(false);
        // aggregation_ = register_module("aggregation_", torch::nn::Conv2d(aggregate));
    }

    torch::Tensor ConvolutionalNeuralNetworkImpl::forward(const torch::Tensor& image)
    {
        auto x = initialization_->forward(image);

        for (uint32_t i = 0; i < layers_; ++i)
        {
            // x and y are computed using residuals, this accelerates convergence significantly
            const auto y = aggregation_->forward(message_->forward(x) + x);
            x = update_->forward(torch::cat({ x, y }, 1)) + x;
        }

        const auto batch_size = x.size(0);
        const auto channels = x.size(1);
        x = x.view({ batch_size, channels, -1 }).sum(2);
        return readout_->forward(x);
    }
}  // namespace models
