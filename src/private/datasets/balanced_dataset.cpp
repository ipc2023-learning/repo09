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


#include "balanced_dataset.hpp"

#include <algorithm>

namespace datasets
{
    BalancedDataset::BalancedDataset(const planners::StateSpaceList& state_spaces,
                                     const bool random_sampling,
                                     const bool remove_dead_ends,
                                     const bool remove_goal_states) :
        state_spaces_(state_spaces),
        samples_(),
        random_sampling_(random_sampling)
    {
        planners::StateSpaceSampleList dead_end_samples;
        std::vector<planners::StateSpaceSampleList> samples_by_goal_distance;

        for (const auto& state_space : state_spaces)
        {
            for (const auto& state : state_space->get_states())
            {
                const auto is_dead_end = state_space->is_dead_end_state(state);

                if (!is_dead_end)
                {
                    const auto is_goal = state_space->is_goal_state(state);

                    if (!remove_goal_states || !is_goal)
                    {
                        const auto goal_distance = state_space->get_distance_to_goal_state(state);
                        if (goal_distance >= (int32_t) samples_by_goal_distance.size())
                        {
                            samples_by_goal_distance.resize(goal_distance + 1);
                        }

                        samples_by_goal_distance[goal_distance].push_back(std::make_pair(state, state_space));
                    }
                }
                else if (!remove_dead_ends && is_dead_end)
                {
                    dead_end_samples.push_back(std::make_pair(state, state_space));
                }
            }
        }

        samples_by_goal_distance.push_back(dead_end_samples);

        std::size_t largest_class = 0;

        for (auto& distance_samples : samples_by_goal_distance)
        {
            largest_class = std::max(largest_class, distance_samples.size());
        }

        auto rng = std::default_random_engine {};
        for (auto& distance_samples : samples_by_goal_distance)
        {
            if (distance_samples.size() > 0)
            {
                std::size_t num_samples = distance_samples.size();
                std::shuffle(distance_samples.begin(), distance_samples.end(), rng);

                while (distance_samples.size() < largest_class)
                {
                    distance_samples.push_back(distance_samples[rng() % num_samples]);
                }
            }
        }

        bool any = true;
        while (any)
        {
            any = false;

            for (auto& distance_samples : samples_by_goal_distance)
            {
                if (distance_samples.size() > 0)
                {
                    any = true;
                    const auto sample = distance_samples.back();
                    distance_samples.pop_back();
                    samples_.push_back(sample);
                }
            }
        }
    }

    uint32_t BalancedDataset::size() const { return samples_.size(); }

    planners::StateSpaceSample BalancedDataset::get(uint32_t index) const
    {
        if (random_sampling_)
        {
            const auto random_index = std::rand() % size();
            return samples_[random_index];
        }

        if (index < size())
        {
            return samples_[index];
        }

        throw std::invalid_argument("index is out of range");
    }

    planners::StateSpaceSampleList BalancedDataset::get_range(uint32_t index, uint32_t count) const
    {
        planners::StateSpaceSampleList states_in_range;

        // ensure that we do not go out of bounds
        if (index + count > size())
        {
            count -= index + count - size();
        }

        for (uint32_t offset = 0; offset < count; ++offset)
        {
            states_in_range.push_back(get(index + offset));
        }

        return states_in_range;
    }

    std::vector<std::pair<std::string, int32_t>> BalancedDataset::get_predicate_name_and_arities() const
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
