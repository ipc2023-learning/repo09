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


#include "../libraries/map_functions.hpp"
#include "../libraries/murmurhash3.hpp"
#include "weisfeiler_leman.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace algorithms
{
    WeisfeilerLeman::WeisfeilerLeman() : k_(1), hash_(), predicate_ids_(), label_ids_() {}

    WeisfeilerLeman::WeisfeilerLeman(int32_t k) : k_(k), hash_(), predicate_ids_(), label_ids_()
    {
        if (k < 1)
        {
            throw std::invalid_argument("k must be at least 1");
        }

        if (k > 1)
        {
            throw std::runtime_error("Current implementation does not support k > 1");
        }
    }

    std::vector<int32_t> WeisfeilerLeman::compute_color_histogram(const WLGraph& graph)
    {
        if (k_ > 1)
        {
            throw std::runtime_error("Current implementation does not support k > 1");
        }

        const auto num_vertices = graph.vertices.size();

        std::vector<int32_t> previous_coloring;
        previous_coloring.resize(num_vertices);  // Default vertex coloring is 0.

        std::vector<int32_t> current_coloring;  // We avoid redefinitions by placing this variable here,
        current_coloring.resize(num_vertices);  // this allow us to avoid unnecessary memory allocations.

        // New colorings are based on the current color and adjacent colors.

        for (std::size_t index = 0; index < num_vertices; ++index)
        {
            std::vector<std::vector<std::pair<int32_t, int32_t>>> vertex_multisets;
            vertex_multisets.resize(num_vertices);

            for (const auto& edge : graph.edges)
            {
                const auto source = std::get<0>(edge);
                const auto target = std::get<1>(edge);
                const auto label = std::get<2>(edge);

                vertex_multisets[source].push_back(std::make_pair(previous_coloring[target], label));

                if (source != target)
                {
                    // Edges should be undirected, this is done by introducing the atom p'(y, x) for each atom p(x, y)
                    vertex_multisets[target].push_back(std::make_pair(previous_coloring[source], -label));
                }
            }

            for (std::size_t index = 0; index < num_vertices; ++index)
            {
                // Order should not matter
                auto& multiset = vertex_multisets[index];
                std::sort(multiset.begin(), multiset.end());

                const auto hash_entry = std::make_pair(previous_coloring[index], multiset);
                const auto hash_handler = hash_.find(hash_entry);

                if (hash_handler != hash_.end())
                {
                    current_coloring[index] = hash_handler->second;
                }
                else
                {
                    const auto hash_value = hash_.size();
                    hash_.insert(std::make_pair(hash_entry, hash_value));
                    current_coloring[index] = hash_value;
                }
            }

            if (previous_coloring == current_coloring)
            {
                // We've reached a fixpoint, thus the result won't change.
                break;
            }

            previous_coloring = current_coloring;
        }

        std::sort(previous_coloring.begin(), previous_coloring.end());
        return previous_coloring;
    }

    std::pair<uint64_t, uint64_t> WeisfeilerLeman::hash_color_histogram(const std::vector<int32_t>& histogram)
    {
        uint64_t hash[2];
        murmurhash3_128(&histogram[0], sizeof(int32_t) * histogram.size(), 0, hash);
        return std::make_pair(hash[0], hash[1]);
    }

    void WeisfeilerLeman::add_atom(const formalism::Atom& atom,
                                   const std::string& predicate_suffix,
                                   std::map<std::string, int32_t>& object_ids,
                                   std::map<std::pair<int32_t, int32_t>, std::vector<int32_t>>& edge_labels,
                                   int32_t& variable_counter)
    {
        const auto predicate = atom->predicate;
        const auto arity = predicate->arity;
        const auto name = predicate->name + predicate_suffix;

        const std::string new_variable_prefix = "internal_new_variable_";
        const std::string new_variable_nullary = new_variable_prefix + "nullary";

        if (arity == 0)
        {
            const int32_t object_id = get_or_add(object_ids, new_variable_nullary, (int32_t) object_ids.size());
            const auto predicate_id = get_or_add(predicate_ids_, name, (int32_t) predicate_ids_.size());
            auto& labels = get_or_add(edge_labels, std::pair(object_id, object_id));
            labels.push_back(predicate_id);
        }
        else if (arity == 1)
        {
            const int32_t object_id = get_or_add(object_ids, atom->arguments[0]->name, (int32_t) object_ids.size());
            const auto predicate_id = get_or_add(predicate_ids_, name, (int32_t) predicate_ids_.size());
            auto& labels = get_or_add(edge_labels, std::pair(object_id, object_id));
            labels.push_back(predicate_id);
        }
        else if (arity == 2)
        {
            const int32_t first_object_id = get_or_add(object_ids, atom->arguments[0]->name, (int32_t) object_ids.size());
            const int32_t second_object_id = get_or_add(object_ids, atom->arguments[1]->name, (int32_t) object_ids.size());
            const auto predicate_id = get_or_add(predicate_ids_, name, (int32_t) predicate_ids_.size());
            auto& labels = get_or_add(edge_labels, std::pair(first_object_id, second_object_id));
            labels.push_back(predicate_id);
        }
        else
        {
            const auto new_variable = new_variable_prefix + std::to_string(variable_counter);
            const auto new_variable_id = get_or_add(object_ids, new_variable, (int32_t) object_ids.size());
            ++variable_counter;

            for (std::size_t index = 0; index < arity; ++index)
            {
                const auto other_object_id = get_or_add(object_ids, atom->arguments[index]->name, (int32_t) object_ids.size());
                const auto predicate_id = get_or_add(predicate_ids_, name + "_" + std::to_string(index), (int32_t) predicate_ids_.size());
                auto& labels = get_or_add(edge_labels, std::pair(new_variable_id, other_object_id));
                labels.push_back(predicate_id);
            }
        }
    }

    WLGraph WeisfeilerLeman::to_wl_graph(const formalism::ProblemDescription& problem, const formalism::State& state)
    {
        std::map<std::string, int32_t> object_ids;
        std::map<std::pair<int32_t, int32_t>, std::vector<int32_t>> edge_labels;
        int32_t variable_counter = 0;

        for (const auto& atom : state->get_atoms())
        {
            add_atom(atom, "", object_ids, edge_labels, variable_counter);
        }

        for (const auto& literal : problem->goal)
        {
            if (literal->negated)
            {
                throw std::invalid_argument("negated goal atoms not supported");
            }

            add_atom(literal->atom, "_goal", object_ids, edge_labels, variable_counter);
        }

        std::set<int32_t> vertices;
        std::vector<std::tuple<int32_t, int32_t, int32_t>> edges;

        for (auto& labeled_edge : edge_labels)
        {
            const auto& edge = labeled_edge.first;
            vertices.insert(edge.first);
            vertices.insert(edge.second);

            auto& labels = labeled_edge.second;
            std::sort(labels.begin(), labels.end());
            const auto label_id = get_or_add(label_ids_, labels, (int32_t) label_ids_.size());
            edges.push_back(std::make_tuple(edge.first, edge.second, label_id));
        }

        return WLGraph(std::vector<int32_t>(vertices.begin(), vertices.end()), edges);
    }

    std::pair<uint64_t, uint64_t> WeisfeilerLeman::compute_state_color(const formalism::ProblemDescription& problem, const formalism::State& state)
    {
        const auto wl_graph = to_wl_graph(problem, state);
        const auto histogram = compute_color_histogram(wl_graph);
        const auto hash = hash_color_histogram(histogram);
        return hash;
    }
}  // namespace algorithms
