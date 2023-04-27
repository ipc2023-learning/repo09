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


// This implementation of MultiheadAttention is based on the Python code from:
// http://einops.rocks/pytorch-examples.html

#include "multihead_attention_module.hpp"

namespace models
{
    MultiheadAttentionImpl::MultiheadAttentionImpl(int32_t input_size, int32_t output_size, int32_t num_heads) :
        input_size_(input_size),
        num_heads_(num_heads),
        linear_query_(nullptr),
        linear_key_(nullptr),
        linear_value_(nullptr),
        linear_final_(nullptr)
    {
        linear_query_ = register_module("linear_query_", torch::nn::Linear(input_size, num_heads * input_size));
        linear_key_ = register_module("linear_key_", torch::nn::Linear(input_size, num_heads * input_size));
        linear_value_ = register_module("linear_value_", torch::nn::Linear(input_size, num_heads * input_size));
        linear_final_ = register_module("linear_final_", torch::nn::Linear(input_size * num_heads, output_size));

        // torch::nn::init::normal_(linear_query_->weight, 0.0, std::sqrt(1.0 / (double) input_size));
        // torch::nn::init::normal_(linear_key_->weight, 0.0, std::sqrt(1.0 / (double) input_size));
        // torch::nn::init::normal_(linear_value_->weight, 0.0, std::sqrt(1.0 / (double) input_size));
        // torch::nn::init::xavier_normal_(linear_final_->weight);
    }

    torch::Tensor MultiheadAttentionImpl::forward(const torch::Tensor& input)
    {
        const auto sizes = input.sizes();
        const auto batch_size = sizes[0];
        const auto seq_length = sizes[1];
        const auto input_size = sizes[2];

        auto query = linear_query_->forward(input).view({ batch_size, seq_length, num_heads_, input_size });
        auto key = linear_key_->forward(input).view({ batch_size, seq_length, num_heads_, input_size });
        auto value = linear_value_->forward(input).view({ batch_size, seq_length, num_heads_, input_size });

        query = query.permute({ 2, 0, 1, 3 }).contiguous().view({ -1, seq_length, input_size });
        key = key.permute({ 2, 0, 1, 3 }).contiguous().view({ -1, seq_length, input_size });
        value = value.permute({ 2, 0, 1, 3 }).contiguous().view({ -1, seq_length, input_size });

        const auto debug = input.bmm(input.transpose(1, 2));

        auto attention = query.bmm(key.transpose(1, 2));
        //         mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        //         if mask is not None:
        //             attn = attn.masked_fill(mask, -np.inf)
        // attention = attention.softmax(2);
        attention = (attention / std::sqrt(input_size)).softmax(2);

        auto output = attention.bmm(value);
        output = output.view({ num_heads_, batch_size, seq_length, input_size });
        output = output.permute({ 1, 2, 0, 3 }).contiguous().view({ batch_size, seq_length, -1 });
        return linear_final_->forward(output);
    }
}  // namespace models
