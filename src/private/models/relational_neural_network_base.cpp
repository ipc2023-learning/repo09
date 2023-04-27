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


#include "relational_neural_network_base.hpp"

namespace models
{
    void add_object_types_to_batch(const formalism::Object& object,
                                   std::map<std::string, std::vector<int64_t>>& batch_predicate_atoms,
                                   const std::map<std::string, int32_t>& object_ids,
                                   const int32_t offset)
    {
        auto current_type = object->type;

        while ((current_type != nullptr) && (current_type->name != "object"))
        {
            int32_t id;
            const auto object_handler = object_ids.find(object->name);

            if (object_handler != object_ids.end())
            {
                id = object_handler->second + offset;
            }
            else
            {
                throw std::runtime_error("Object is not in object_ids");
            }

            const auto type_name = current_type->name;
            auto batch_predicate_handler = batch_predicate_atoms.find(type_name);

            if (batch_predicate_handler != batch_predicate_atoms.end())
            {
                auto& batch_type = batch_predicate_handler->second;
                batch_type.push_back(id);
            }
            else
            {
                batch_predicate_atoms.insert(std::make_pair(type_name, std::vector<int64_t>({ id })));
            }

            current_type = current_type->base;
        }
    }

    void add_atom_to_batch(const formalism::Atom& atom,
                           std::map<std::string, std::vector<int64_t>>& batch_predicate_atoms,
                           const std::map<std::string, int32_t>& object_ids,
                           const int32_t offset,
                           const std::string& suffix)
    {
        std::vector<int64_t> batch_arguments;

        for (const auto& object : atom->arguments)
        {
            // Get the unique id of the object.

            const auto object_handler = object_ids.find(object->name);
            int32_t id;

            if (object_handler != object_ids.end())
            {
                id = object_handler->second;
            }
            else
            {
                throw std::runtime_error("Object is not in object_ids");
            }

            batch_arguments.push_back(id + offset);
        }

        // Add all arguments to the same predicate vector using an id unique to the state.
        const auto atom_name = atom->predicate->name + suffix;
        auto batch_atoms_handler = batch_predicate_atoms.find(atom_name);

        if (batch_atoms_handler != batch_predicate_atoms.end())
        {
            auto& batch_atoms = batch_atoms_handler->second;
            batch_atoms.insert(batch_atoms.end(), batch_arguments.begin(), batch_arguments.end());
        }
        else
        {
            batch_predicate_atoms.insert(std::make_pair(atom_name, batch_arguments));
        }
    }

    void add_derived_predicates_to_batch(std::map<std::string, std::vector<int64_t>>& batch_predicate_atoms,
                                         std::vector<int64_t>& batch_sizes,
                                         int32_t& offset,
                                         const std::map<std::string, int32_t>& object_ids,
                                         const formalism::ProblemDescription& problem,
                                         const formalism::State& state,
                                         const models::InternalDerivedPredicateList& derived_predicates)
    {
        if (derived_predicates.size() == 0)
        {
            return;
        }

        std::map<std::string, formalism::Predicate> name_to_predicate;
        auto state_atoms = state->get_atoms_grouped_by_predicate();
        std::map<formalism::Predicate, std::set<formalism::ObjectList>> added_derived_atoms;
        const auto& domain_predicates = problem->domain->predicates;

        for (const auto& predicate : domain_predicates)
        {
            name_to_predicate.insert(std::make_pair(predicate->name, predicate));

            if (state_atoms.find(predicate) == state_atoms.end())
            {
                state_atoms.insert(std::make_pair(predicate, formalism::AtomList()));
            }
        }

        // We need to initialize empty lists of atoms for derived predicates to the state.

        const auto num_predicates = domain_predicates.size();
        const std::vector<std::string> param_names({ "?x", "?y", "?z", "?w", "?n", "?m" });

        for (std::size_t index = 0; index < derived_predicates.size(); ++index)
        {
            const auto& derived_predicate = derived_predicates[index];
            const auto& predicate_name = std::get<0>(derived_predicate);
            const auto& num_bound_variables = std::get<1>(derived_predicate);

            formalism::ObjectList params;
            for (int32_t param_index = 0; param_index < num_bound_variables; ++param_index)
            {
                params.push_back(formalism::create_object(param_index, param_names.at(param_index), formalism::create_type("object")));
            }

            const auto predicate = formalism::create_predicate(2 * num_predicates + index, predicate_name, params);
            name_to_predicate.insert(std::make_pair(predicate_name, predicate));
            state_atoms.insert(std::make_pair(predicate, formalism::AtomList()));
            added_derived_atoms.insert(std::make_pair(predicate, std::set<formalism::ObjectList>()));
        }

        // Add goals with the suffix _goal to the state

        for (const auto& predicate : domain_predicates)
        {
            const auto predicate_goal_name = predicate->name + "_goal";
            const auto goal_predicate = formalism::create_predicate(num_predicates + predicate->id, predicate_goal_name, predicate->parameters);
            name_to_predicate.insert(std::make_pair(predicate_goal_name, goal_predicate));
            state_atoms.insert(std::make_pair(goal_predicate, formalism::AtomList()));
        }

        for (const auto& literal : problem->goal)
        {
            if (!literal->negated)
            {
                const auto& goal_predicate = name_to_predicate.at(literal->atom->predicate->name + "_goal");
                auto& goal_atoms = state_atoms.at(goal_predicate);
                goal_atoms.push_back(formalism::create_atom(goal_predicate, literal->atom->arguments));
            }
        }

        // Loop until we have not added any new derived atom to the state.

        bool any_change = true;

        while (any_change)
        {
            any_change = false;

            for (const auto& dp_tuple : derived_predicates)
            {
                const auto& derived_predicate_name = std::get<0>(dp_tuple);
                const auto& derived_predicate = name_to_predicate.at(derived_predicate_name);
                const auto& num_bound_variables = std::get<1>(dp_tuple);
                const auto& num_variables = std::get<2>(dp_tuple);
                const auto& case_list = std::get<3>(dp_tuple);

                for (const auto& case_atoms : case_list)
                {
                    std::vector<std::vector<formalism::Object>> groundings;
                    std::vector<std::vector<formalism::Object>> next_groundings;
                    groundings.push_back(std::vector<formalism::Object>(num_variables, nullptr));

                    for (std::size_t atom_index = 0; atom_index < case_atoms.size(); ++atom_index)
                    {
                        const auto& predicate_name = case_atoms[atom_index].first;
                        const auto& predicate_args = case_atoms[atom_index].second;
                        const auto& predicate = name_to_predicate.at(predicate_name);
                        const auto& predicate_state_atoms = state_atoms.at(predicate);

                        for (const auto& grounding : groundings)
                        {
                            for (const auto& state_atom : predicate_state_atoms)
                            {
                                bool consistent_assignment = true;
                                auto new_grounding = grounding;
                                const auto& atom_args = state_atom->arguments;

                                for (std::size_t arg_index = 0; arg_index < atom_args.size(); ++arg_index)
                                {
                                    const auto predicate_arg = predicate_args[arg_index];
                                    const auto atom_arg = atom_args[arg_index];

                                    if (new_grounding[predicate_arg] == nullptr)
                                    {
                                        new_grounding[predicate_arg] = atom_arg;
                                    }
                                    else if (new_grounding[predicate_arg] != atom_arg)
                                    {
                                        consistent_assignment = false;
                                        break;
                                    }
                                }

                                if (consistent_assignment)
                                {
                                    next_groundings.push_back(std::move(new_grounding));
                                }
                            }
                        }

                        groundings.clear();
                        std::swap(groundings, next_groundings);
                    }

                    int32_t num_derived_atoms = 0;

                    for (const auto& combination : groundings)
                    {
                        formalism::ObjectList args(combination.begin(), combination.begin() + num_bound_variables);
                        auto& predicate_duplicates = added_derived_atoms.at(derived_predicate);

                        if (predicate_duplicates.find(args) == predicate_duplicates.end())
                        {
                            ++num_derived_atoms;
                            predicate_duplicates.insert(args);
                            auto& atoms = state_atoms.at(derived_predicate);
                            atoms.push_back(formalism::create_atom(derived_predicate, args));
                        }
                    }

                    if (num_derived_atoms > 0)
                    {
                        any_change = true;
                    }
                }
            }
        }

        for (const auto& derived_predicate : derived_predicates)
        {
            const auto& derived_predicate_name = std::get<0>(derived_predicate);
            const auto& der_predicate = name_to_predicate.at(derived_predicate_name);

            const auto& atoms = state_atoms.at(der_predicate);

            for (const auto& atom : atoms)
            {
                add_atom_to_batch(atom, batch_predicate_atoms, object_ids, offset, "");
            }
        }
    }

    void add_atoms_and_goal_to_batch(std::map<std::string, std::vector<int64_t>>& batch_predicate_atoms,
                                     std::vector<int64_t>& batch_sizes,
                                     int32_t& offset,
                                     const std::map<std::string, int32_t>& object_ids,
                                     const formalism::ProblemDescription& problem,
                                     const formalism::State& state,
                                     const models::InternalDerivedPredicateList& derived_predicates)
    {
        for (const auto& object : problem->objects)
        {
            add_object_types_to_batch(object, batch_predicate_atoms, object_ids, offset);
        }

        for (const auto& atom : state->get_atoms())
        {
            add_atom_to_batch(atom, batch_predicate_atoms, object_ids, offset, "");
        }

        for (const auto& literal : problem->goal)
        {
            if (literal->negated)
            {
                throw std::invalid_argument("negative literal in goal");
            }

            add_atom_to_batch(literal->atom, batch_predicate_atoms, object_ids, offset, "_goal");
        }

        add_derived_predicates_to_batch(batch_predicate_atoms, batch_sizes, offset, object_ids, problem, state, derived_predicates);

        batch_sizes.push_back(object_ids.size());
        offset += object_ids.size();
    }

    RelationalNeuralNetworkBase::RelationalNeuralNetworkBase() :
        predicate_arities_(),
        predicate_ids_(),
        id_arities_(),
        external_derived_predicates_(),
        internal_derived_predicates_(),
        dummy_(torch::empty(0))
    {
        dummy_ = register_parameter("dummy_", dummy_, false);
    }

    RelationalNeuralNetworkBase::RelationalNeuralNetworkBase(const PredicateArityList& predicates, const DerivedPredicateList& derived_predicates) :
        predicate_arities_(predicates.begin(), predicates.end()),
        predicate_ids_(),
        id_arities_(),
        external_derived_predicates_(derived_predicates.begin(), derived_predicates.end()),
        internal_derived_predicates_(),
        dummy_(torch::empty(0))
    {
        // TODO: This is needed for Transformer, which is broken atm.
        // for (const auto& predicate_arity : predicates)
        // {
        //     const auto& name = predicate_arity.first;
        //     const auto& arity = predicate_arity.second;
        //     predicate_arities_.insert(std::make_pair(name, arity));
        //     predicate_arities_.insert(std::make_pair(name + "_goal", arity));
        // }

        // Sort predicates so that no matter the order, we get the same assignment of ids.
        auto sorted_predicates = predicates;
        std::sort(sorted_predicates.begin(), sorted_predicates.end());

        for (const auto& predicate_arity : sorted_predicates)
        {
            const auto predicate = predicate_arity.first;
            const auto arity = predicate_arity.second;

            if (arity > 0)
            {
                const auto id = predicate_ids_.size();
                predicate_ids_.insert(std::make_pair(predicate, id));
                predicate_ids_.insert(std::make_pair(predicate + "_goal", id + 1));
                id_arities_.push_back(std::make_pair(id, arity));
                id_arities_.push_back(std::make_pair(id + 1, arity));
            }
        }

        // Process derived predicates to the internal format.
        auto sorted_derived_predicates = derived_predicates;
        std::sort(sorted_derived_predicates.begin(), sorted_derived_predicates.end());

        for (const auto& derived_predicate : derived_predicates)
        {
            const auto predicate = derived_predicate.first.first;
            const auto params = derived_predicate.first.second;

            if (params.size() == 0)
            {
                throw std::runtime_error("arity of derived predicate is 0");
            }

            const auto predicate_id = predicate_ids_.size();
            predicate_ids_.insert(std::make_pair(predicate, predicate_id));
            id_arities_.push_back(std::make_pair(predicate_id, params.size()));
        }

        for (const auto& derived_predicate : derived_predicates)
        {
            const auto predicate_name = derived_predicate.first.first;
            const auto params = derived_predicate.first.second;
            const auto body_case_list = derived_predicate.second;
            int32_t max_variables = -1;

            std::map<std::string, int32_t> param_ids;
            for (const auto& param : params)
            {
                param_ids.insert(std::pair(param, param_ids.size()));
            }

            std::vector<std::vector<std::pair<std::string, std::vector<int32_t>>>> internal_case_list;
            for (const auto& body_case : body_case_list)
            {
                std::vector<std::pair<std::string, std::vector<int32_t>>> internal_case;

                for (const auto& body_case_predicate : body_case)
                {
                    const auto& body_predicate_name = body_case_predicate.first;
                    const auto& body_params = body_case_predicate.second;
                    std::vector<int32_t> body_param_ids;

                    for (const auto& body_param : body_params)
                    {
                        int32_t body_param_id;
                        const auto body_param_id_handler = param_ids.find(body_param);

                        if (body_param_id_handler == param_ids.end())
                        {
                            body_param_id = param_ids.size();
                            param_ids.insert(std::make_pair(body_param, body_param_id));
                        }
                        else
                        {
                            body_param_id = body_param_id_handler->second;
                        }

                        body_param_ids.push_back(body_param_id);
                    }

                    internal_case.push_back(std::make_pair(body_predicate_name, body_param_ids));
                }

                max_variables = std::max(max_variables, (int32_t) param_ids.size());
                internal_case_list.push_back(std::move(internal_case));
            }

            internal_derived_predicates_.push_back(std::make_tuple(predicate_name, params.size(), max_variables, internal_case_list));
        }

        dummy_ = register_parameter("dummy_", dummy_, false);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RelationalNeuralNetworkBase::forward(const formalism::StateTransitions& state_transitions)
    {
        const auto result = forward(formalism::StateTransitionsVector({ state_transitions }));
        return std::make_tuple(std::get<0>(result).at(0), std::get<1>(result).at(0), std::get<2>(result).at(0));
    }

    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
    RelationalNeuralNetworkBase::forward(const formalism::StateTransitionsVector& state_transitions)
    {
        std::map<std::string, std::vector<int64_t>> batch_predicate_atoms;
        std::vector<int64_t> batch_sizes;
        int32_t offset = 0;

        std::vector<std::tuple<int32_t, int32_t, int32_t>> batch_slices;
        int32_t slice_offset = 0;

        std::map<formalism::ProblemDescription, std::map<std::string, int32_t>> problem_object_ids;

        for (const auto& transition : state_transitions)
        {
            const auto& problem = std::get<2>(transition);

            if (problem_object_ids.find(problem) == problem_object_ids.end())
            {
                std::map<std::string, int32_t> object_ids;

                for (const auto& object : problem->objects)
                {
                    object_ids.insert(std::make_pair(object->name, (int32_t) object_ids.size()));
                }

                problem_object_ids.insert(std::make_pair(problem, std::move(object_ids)));
            }
        }

        for (const auto& transition : state_transitions)
        {
            const auto& state = std::get<0>(transition);
            const auto& successors = std::get<1>(transition);
            const auto& problem = std::get<2>(transition);
            const auto slice_start = slice_offset;
            const auto slice_end = slice_start + successors.size() + 1;
            slice_offset = slice_end;

            const auto& objects = problem->objects;
            batch_slices.push_back(std::make_tuple(slice_start, slice_end, objects.size()));

            const auto& object_ids = problem_object_ids.at(problem);
            const auto atoms = state->get_atoms();

            add_atoms_and_goal_to_batch(batch_predicate_atoms, batch_sizes, offset, object_ids, problem, state, internal_derived_predicates_);

            for (const auto& successor : successors)
            {
                add_atoms_and_goal_to_batch(batch_predicate_atoms, batch_sizes, offset, object_ids, problem, successor, internal_derived_predicates_);
            }
        }

        const auto object_embeddings = internal_forward(batch_predicate_atoms, batch_sizes);
        const auto policy_vector = readout_policy(object_embeddings, batch_sizes, batch_slices);
        const auto value_vector = readout_value(object_embeddings, batch_sizes, batch_slices);
        const auto dead_end_vector = readout_dead_end(object_embeddings, batch_sizes, batch_slices);

        return std::make_tuple(policy_vector, value_vector, dead_end_vector);
    }

    std::pair<torch::Tensor, torch::Tensor> RelationalNeuralNetworkBase::forward(const formalism::StateProblemList& state_problems)
    {
        std::map<std::string, std::vector<int64_t>> batch_predicate_atoms;
        std::vector<int64_t> batch_sizes;
        int32_t offset = 0;

        for (const auto& state_problem : state_problems)
        {
            const auto& state = state_problem.first;
            const auto& problem = state_problem.second;

            std::map<std::string, int32_t> object_ids;
            for (const auto& object : problem->objects)
            {
                object_ids.insert(std::make_pair(object->name, (int32_t) object_ids.size()));
            }

            add_atoms_and_goal_to_batch(batch_predicate_atoms, batch_sizes, offset, object_ids, problem, state, internal_derived_predicates_);
        }

        const auto object_embeddings = internal_forward(batch_predicate_atoms, batch_sizes);
        const auto values = readout_value(object_embeddings, batch_sizes);
        const auto dead_ends = readout_dead_end(object_embeddings, batch_sizes);

        return std::make_pair(values, dead_ends);
    }

    std::pair<torch::Tensor, torch::Tensor> RelationalNeuralNetworkBase::forward(const formalism::StateList& states,
                                                                                 const formalism::ProblemDescription& problem)
    {
        std::map<std::string, std::vector<int64_t>> batch_predicate_atoms;
        std::vector<int64_t> batch_sizes;
        int32_t offset = 0;

        std::map<std::string, int32_t> object_ids;
        for (const auto& object : problem->objects)
        {
            object_ids.insert(std::make_pair(object->name, (int32_t) object_ids.size()));
        }

        for (const auto& state : states)
        {
            add_atoms_and_goal_to_batch(batch_predicate_atoms, batch_sizes, offset, object_ids, problem, state, internal_derived_predicates_);
        }

        const auto object_embeddings = internal_forward(batch_predicate_atoms, batch_sizes);
        const auto values = readout_value(object_embeddings, batch_sizes);
        const auto dead_ends = readout_dead_end(object_embeddings, batch_sizes);

        return std::make_pair(values, dead_ends);
    }

    std::pair<torch::Tensor, torch::Tensor>
    RelationalNeuralNetworkBase::forward(const formalism::StateList& states, const formalism::ProblemDescription& problem, uint32_t chunk_size)
    {
        const uint32_t num_states = (uint32_t) states.size();
        const uint32_t num_chunks = (num_states / chunk_size) + ((num_states % chunk_size) > 0 ? 1 : 0);

        auto values = torch::zeros({ 0, 1 }).to(device());
        auto dead_ends = torch::zeros({ 0, 1 }).to(device());

        for (uint32_t chunk_index = 0; chunk_index < num_chunks; ++chunk_index)
        {
            auto first = states.begin() + chunk_index * chunk_size;
            auto last = states.begin() + std::min(num_states, chunk_index * chunk_size + chunk_size);
            formalism::StateList chunk(first, last);

            const auto output = forward(chunk, problem);
            values = torch::cat({ values, output.first }, 0);
            dead_ends = torch::cat({ dead_ends, output.second }, 0);
        }

        return std::make_pair(values, dead_ends);
    }
}  // namespace models
