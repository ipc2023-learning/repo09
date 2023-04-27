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


#include "reduce_lr_on_plateau.hpp"

#include <limits>

namespace experiments
{
    ReduceLROnPlateau::ReduceLROnPlateau(torch::optim::Optimizer& optimizer, double factor, uint32_t patience, double min_learning_rate) :
        optimizer_(optimizer),
        factor_(factor),
        best_loss_(std::numeric_limits<double>::infinity()),
        min_learning_rate_(min_learning_rate),
        patience_(patience),
        plateau_counter_(0),
        callback_(nullptr)
    {
    }

    void ReduceLROnPlateau::decrease_learning_rate()
    {
        if (optimizer_.param_groups().size() == 0)
        {
            return;
        }

        for (auto& param_group : optimizer_.param_groups())
        {
            if (param_group.has_options())
            {
                auto& options = param_group.options();
                double current_learning_rate = options.get_lr();
                double new_learning_rate = std::max(min_learning_rate_, current_learning_rate * factor_);

                options.set_lr(new_learning_rate);

                if (callback_)
                {
                    callback_(new_learning_rate);
                }

                break;
            }
        }
    }

    void ReduceLROnPlateau::update(double loss)
    {
        if (loss < best_loss_)
        {
            best_loss_ = loss;
            plateau_counter_ = 0;
        }
        else
        {
            ++plateau_counter_;
        }

        if (plateau_counter_ >= patience_)
        {
            decrease_learning_rate();
            plateau_counter_ = 0;
        }
    }

    void ReduceLROnPlateau::register_on_lr_update(CallbackType callback) { callback_ = callback; }
}  // namespace experiments
