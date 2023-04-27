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


#include "search_base.hpp"

#include <algorithm>

namespace planners
{
    SearchStateRepository::SearchStateRepository() : states_(), indices_(), contexts_() {}

    bool SearchStateRepository::add_or_get_state(const formalism::State& state, uint32_t& out_state_index)
    {
        const auto index_handler = indices_.find(state);

        if (index_handler != indices_.end())
        {
            out_state_index = index_handler->second;
            return false;
        }
        else
        {
            out_state_index = contexts_.size();
            states_.push_back(state);
            contexts_.push_back({ -1.0, -1.0, nullptr, -1U });
            indices_.insert(std::make_pair(state, out_state_index));
            return true;
        }
    }

    StateContext& SearchStateRepository::get_context(uint32_t state_index) { return contexts_[state_index]; }

    formalism::State& SearchStateRepository::get_state(uint32_t state_index) { return states_[state_index]; }

    uint32_t SearchStateRepository::num_indices() const { return static_cast<uint32_t>(states_.size()); }

    std::vector<formalism::Action> SearchStateRepository::get_plan(uint32_t state_index)
    {
        std::vector<formalism::Action> plan;

        auto current_state_index = state_index;

        while (true)
        {
            const auto& current_context = contexts_[current_state_index];
            const auto predecessor_action = current_context.predecessor_action;

            if (predecessor_action == nullptr)
            {
                break;
            }

            plan.push_back(predecessor_action);
            current_state_index = current_context.predecessor_state_index;
        }

        std::reverse(plan.begin(), plan.end());

        return plan;
    }

    SearchBase::SearchBase() : repository() {}

}  // namespace planners
