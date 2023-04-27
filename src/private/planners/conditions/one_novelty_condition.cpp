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


#include "one_novelty_condition.hpp"

namespace planners
{
    OneNoveltyCondition::OneNoveltyCondition(const formalism::ProblemDescription& problem) : problem_(problem), seen_(problem->num_ranks(), false) {}

    // Returns true if all atoms has been seen before, otherwise false
    bool OneNoveltyCondition::test(const formalism::State& state)
    {
        bool novel = false;

        for (const auto rank : state->get_ranks())
        {
            if (!seen_[rank])
            {
                seen_[rank] = true;
                novel = true;
            }
        }

        return !novel;
    }
}  // namespace planners
