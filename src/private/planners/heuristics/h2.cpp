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


#include "../generators/lifted_schema_successor_generator.hpp"
#include "h2.hpp"

#include <algorithm>
#include <deque>
#include <iterator>
#include <limits>

namespace planners
{
    int32_t H2Heuristic::get_id(const formalism::Atom& atom) const
    {
        const auto atom_ids_handler = atom_ids_.find(atom);

        if (atom_ids_handler != atom_ids_.end())
        {
            return atom_ids_handler->second;
        }
        else
        {
            int32_t new_id = atom_ids_.size();
            atom_ids_.insert(std::make_pair(atom, new_id));
            return new_id;
        }
    }

    H2Heuristic::H2Heuristic(const formalism::ProblemDescription& problem)
    {
        // Create ids for all dynamic and static atoms, and represent the goal as a list of ids.

        // TODO: Is it sufficient to only consider atoms in the initial state?! (should we do get_rank instead?)

        for (const auto& atom : problem->initial->get_atoms())
        {
            get_id(atom);
        }

        for (const auto& literal : problem->goal)
        {
            if (literal->negated)
            {
                throw std::invalid_argument("[h2] negated goal atoms are not supported");
            }

            goal_.push_back(get_id(literal->atom));
        }

        // Represent actions as STRIPS actions, where each component is represented as a list of ids.

        for (const auto& action_schema : problem->domain->action_schemas)
        {
            const auto action_generator = planners::LiftedSchemaSuccessorGenerator(action_schema, problem);  // TODO: Replace with non-schema version.
            const auto external_actions = action_generator.get_actions();

            for (const auto& action : external_actions)
            {
                H2Action internal_action;

                for (const auto& literal : action->get_precondition())
                {
                    if (literal->negated)
                    {
                        throw std::invalid_argument("[h2] negated precondition atoms are not supported");
                    }

                    internal_action.preconditions.push_back(get_id(literal->atom));
                }

                for (const auto& literal : action->get_effect())
                {
                    if (literal->negated)
                    {
                        internal_action.deletes.push_back(get_id(literal->atom));
                    }
                    else
                    {
                        internal_action.adds.push_back(get_id(literal->atom));
                    }
                }

                internal_action.cost = action->cost;
                actions_.push_back(internal_action);
            }
        }

        for (auto& internal_action : actions_)
        {
            const auto num_ids = static_cast<int32_t>(atom_ids_.size());

            for (int32_t id = 0; id < num_ids; ++id)
            {
                if (std::find(internal_action.deletes.begin(), internal_action.deletes.end(), id) == internal_action.deletes.end())
                {
                    internal_action.deletesComplement.push_back(id);
                }
            }
        }

        // Pre-allocate memory for ht1 and ht2.

        const auto num_state_variables = atom_ids_.size();
        ht1_.resize(num_state_variables);
        ht2_.resize(num_state_variables);

        for (std::size_t i = 0; i < num_state_variables; ++i)
        {
            ht2_[i].resize(num_state_variables);
        }

        // Additive h2 can be significantly more informative, but for now we do not partition actions.

        const auto num_actions = actions_.size();
        std::vector<bool> all_actions_partition;
        all_actions_partition.resize(num_actions);
        std::fill(all_actions_partition.begin(), all_actions_partition.end(), true);
        partitions_.push_back(all_actions_partition);
    }

    int32_t H2Heuristic::evaluate(const std::vector<int32_t>& s) const
    {
        int32_t v = 0;

        for (std::size_t i = 0; i < s.size(); i++)
        {
            v = std::max(v, ht1_[s[i]]);

            if (v == INTERNAL_DEAD_END)
            {
                return INTERNAL_DEAD_END;
            }

            for (std::size_t j = i + 1; j < s.size(); j++)
            {
                v = std::max(v, ht2_[s[i]][s[j]]);

                if (v == INTERNAL_DEAD_END)
                {
                    return INTERNAL_DEAD_END;
                }
            }
        }

        return v;
    }

    int32_t H2Heuristic::evaluate(const std::vector<int32_t>& s, int32_t x) const
    {
        int32_t v = 0;

        v = std::max(v, ht1_[x]);

        if (v == INTERNAL_DEAD_END)
        {
            return INTERNAL_DEAD_END;
        }

        for (std::size_t i = 0; i < s.size(); i++)
        {
            if (x == s[i])
            {
                continue;
            }

            v = std::max(v, ht2_[x][s[i]]);

            if (v == INTERNAL_DEAD_END)
            {
                return INTERNAL_DEAD_END;
            }
        }

        return v;
    }

    void H2Heuristic::update(const std::size_t val, const int32_t h, bool& changed) const
    {
        if (ht1_[val] > h)
        {
            ht1_[val] = h;
            changed = true;
        }
    }

    void H2Heuristic::update(const std::size_t val1, const std::size_t val2, const int32_t h, bool& changed) const
    {
        if (ht2_[val1][val2] > h)
        {
            ht2_[val1][val2] = h;
            ht2_[val2][val1] = h;
            changed = true;
        }
    }

    void H2Heuristic::fill_tables(const std::vector<bool>& partition, const formalism::State& state) const
    {
        const auto num_atoms = ht1_.size();

        for (std::size_t i = 0; i < num_atoms; ++i)
        {
            ht1_[i] = INTERNAL_DEAD_END;

            for (std::size_t j = 0; j < num_atoms; ++j)
            {
                ht2_[i][j] = INTERNAL_DEAD_END;
            }
        }

        const auto& state_atoms = state->get_atoms();
        std::vector<int32_t> state_ids;

        for (const auto& atom : state_atoms)
        {
            const auto atom_ids_handler = atom_ids_.find(atom);

            if (atom_ids_handler != atom_ids_.end())
            {
                state_ids.push_back(atom_ids_handler->second);
            }
            else
            {
                throw std::runtime_error("[h2] internal error, no id associated with atom");
            }
        }

        for (std::size_t i = 0; i < state_ids.size(); ++i)
        {
            ht1_[state_ids[i]] = 0;

            for (std::size_t j = 0; j < state_ids.size(); ++j)
            {
                ht2_[state_ids[i]][state_ids[j]] = 0;
            }
        }

        bool changed;

        do
        {
            changed = false;

            for (std::size_t action_index = 0; action_index < actions_.size(); ++action_index)
            {
                const auto& action = actions_[action_index];
                const auto action_cost = partition[action_index] ? action.cost : 0;
                const auto c1 = evaluate(action.preconditions);

                if (c1 == INTERNAL_DEAD_END)
                {
                    continue;
                }

                for (std::size_t i = 0; i < action.adds.size(); i++)
                {
                    const auto p = action.adds[i];
                    update(p, c1 + action_cost, changed);

                    for (std::size_t j = i + 1; j < action.adds.size(); j++)
                    {
                        const auto q = action.adds[j];

                        if (p != q)
                        {
                            update(p, q, c1 + action_cost, changed);
                        }
                    }

                    for (const auto r : action.deletesComplement)
                    {
                        const auto c2 = std::max(c1, evaluate(action.preconditions, r));

                        if (c2 != INTERNAL_DEAD_END)
                        {
                            update(p, r, c2 + action_cost, changed);
                        }
                    }
                }
            }
        } while (changed);
    }

    double H2Heuristic::get_cost(const formalism::State& state) const
    {
        double heuristic_value = 0.0;

        for (const auto& partition : partitions_)
        {
            fill_tables(partition, state);
            const auto heuristic_term = evaluate(goal_);

            if (heuristic_term == INTERNAL_DEAD_END)
            {
                return DEAD_END;
            }

            heuristic_value += static_cast<double>(heuristic_term);
        }

        return heuristic_value;
    }
}  // namespace planners
