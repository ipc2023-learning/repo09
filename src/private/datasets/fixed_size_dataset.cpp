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


#include "fixed_size_dataset.hpp"

#include <algorithm>

namespace datasets
{
    FixedSizeBalancedDataset::FixedSizeBalancedDataset(const planners::StateSpaceList& state_spaces, uint32_t size) :
        state_spaces_(state_spaces),
        states_grouped_by_label_(),
        generator_(std::random_device {}()),
        size_(size)
    {
        states_grouped_by_label_.resize(state_spaces.size());

        for (std::size_t state_space_index = 0; state_space_index < state_spaces_.size(); ++state_space_index)
        {
            const auto& state_space = state_spaces_[state_space_index];
            auto& map = states_grouped_by_label_[state_space_index];

            for (const auto& state : state_space->get_states())
            {
                const auto label = state_space->get_distance_to_goal_state(state);
                auto& label_group = map[label];
                label_group.emplace_back(state);
            }
        }
    }

    uint32_t FixedSizeBalancedDataset::size() const { return size_; }

    planners::StateSpaceSample FixedSizeBalancedDataset::get(uint32_t index) const
    {
        std::uniform_int_distribution<std::size_t> state_space_uniform(0, state_spaces_.size() - 1);
        const auto state_space_index = state_space_uniform(generator_);
        const auto& state_space = state_spaces_[state_space_index];
        const auto& label_groups = states_grouped_by_label_[state_space_index];


        std::uniform_int_distribution<std::size_t> label_groups_uniform(0, label_groups.size() - 1);
        const auto group_label = label_groups_uniform(generator_);

        const auto dead_end_label = std::numeric_limits<uint32_t>::max();
        const auto selected_dead_end_group = (group_label == (label_groups.size() - 1)) && label_groups.count(dead_end_label);
        const auto& group = selected_dead_end_group ? label_groups.at(dead_end_label) : label_groups.at(group_label);

        std::uniform_int_distribution<std::size_t> group_uniform(0, group.size() - 1);
        const auto state_index = group_uniform(generator_);
        const auto& state = group[state_index];

        return std::make_pair(state, state_space);
    }

    planners::StateSpaceSampleList FixedSizeBalancedDataset::get_range(uint32_t index, uint32_t count) const
    {
        planners::StateSpaceSampleList batch;

        // ensure that we do not go out of bounds
        if (index + count > size())
        {
            count -= index + count - size();
        }

        for (uint32_t offset = 0; offset < count; ++offset)
        {
            batch.push_back(get(index + offset));
        }

        return batch;
    }

    std::vector<std::pair<std::string, int32_t>> FixedSizeBalancedDataset::get_predicate_name_and_arities() const
    {
        if (state_spaces_.size() > 0)
        {
            std::vector<std::pair<std::string, int32_t>> predicate_name_and_arities;
            const auto& predicates = state_spaces_[0]->domain->predicates;

            for (const auto& predicate : predicates)
            {
                predicate_name_and_arities.push_back(std::make_pair(predicate->name, predicate->arity));
            }

            return predicate_name_and_arities;
        }
        else
        {
            throw std::invalid_argument("dataset is empty");
        }
    }

}  // namespace datasets
