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


#include "cosine_annealing_lr.hpp"
#include "torch/torch.h"

#include <iostream>

namespace experiments
{
    CosineAnnealingLR::CosineAnnealingLR(torch::optim::Optimizer& optimizer, size_t max_epochs, double min_learning_rate) :
        optimizer_(optimizer),
        max_epochs_(max_epochs),
        min_learning_rate_(min_learning_rate),
        callback_(nullptr)
    {
    }

    void CosineAnnealingLR::step(size_t current_epoch)
    {
        for (auto& param_group : optimizer_.param_groups())
        {
            if (param_group.has_options())
            {
                auto& options = param_group.options();
                double initial_learning_rate = options.get_lr();
                const double pi = 3.14159265358979323846;
                double new_learning_rate =
                    min_learning_rate_ + (initial_learning_rate - min_learning_rate_) * 0.5 * (1 + std::cos((current_epoch * pi) / max_epochs_));

                options.set_lr(new_learning_rate);

                if (callback_)
                {
                    callback_(new_learning_rate);
                }

                break;
            }
        }
    }

    void CosineAnnealingLR::on_lr_update(OnLearningRateUpdateCallback callback) { callback_ = callback; }
}  // namespace experiments
