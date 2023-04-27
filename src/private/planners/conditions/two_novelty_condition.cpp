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


#include "two_novelty_condition.hpp"

#include <cassert>

namespace planners
{
    TwoNoveltyCondition::TwoNoveltyCondition(const formalism::ProblemDescription& problem) : problem_(problem), seen_() {}

    // Returns true if all atom pairs has been seen before, otherwise false
    bool TwoNoveltyCondition::test(const formalism::State& state)
    {
        bool novel = false;
        const auto ranks = state->get_ranks();

        for (std::size_t i = 0; i < ranks.size(); ++i)
        {
            const auto first_rank = ranks[i];

            for (std::size_t j = i + 1; j < ranks.size(); ++j)
            {
                const auto second_rank = ranks[j];
                assert(first_rank < second_rank);

                if (seen_.insert(std::make_pair(first_rank, second_rank)).second)
                {
                    novel = true;
                }
            }
        }

        return !novel;
    }
}  // namespace planners
