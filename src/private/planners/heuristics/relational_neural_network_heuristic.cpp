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


#include "relational_neural_network_heuristic.hpp"

namespace planners
{
    RelationalNeuralNetworkHeuristic::RelationalNeuralNetworkHeuristic(const formalism::ProblemDescription& problem,
                                                                       const models::RelationalNeuralNetwork& model,
                                                                       uint32_t chunk_size) :
        problem_(problem),
        model_(model),
        cache_(),
        chunk_size_(chunk_size)
    {
    }

    double RelationalNeuralNetworkHeuristic::get_cost(const formalism::State& state) const
    {
        if (cache_.count(state))
        {
            return cache_.at(state);
        }

        const auto& [values, dead_ends] = model_.forward({ state }, problem_);
        const auto heuristics_gpu = values / (1.0 - (dead_ends.sigmoid() - 1E-4).clamp_min(0.0));
        const auto cost = heuristics_gpu[0].item<double>();
        cache_.insert(std::make_pair(state, cost));

        return cost;
    }

    std::vector<double> RelationalNeuralNetworkHeuristic::get_cost(const std::vector<formalism::State>& states) const
    {
        std::vector<formalism::State> batch_states;
        std::vector<std::size_t> batch_indices;

        std::vector<double> costs;
        costs.resize(states.size());

        for (std::size_t state_index = 0; state_index < states.size(); ++state_index)
        {
            const auto& state = states[state_index];

            if (cache_.count(state))
            {
                costs[state_index] = cache_.at(state);
            }
            else
            {
                batch_states.emplace_back(state);
                batch_indices.emplace_back(state_index);
            }
        }

        if (batch_states.size() > 0)
        {
            const auto& [values, dead_ends] = model_.forward(batch_states, problem_, chunk_size_);
            const auto heuristics_gpu = values.view(-1) / (1.0 - (dead_ends.view(-1).sigmoid() - 1E-4).clamp_min(0.0));
            const auto heuristics_cpu = heuristics_gpu.cpu();

            if (heuristics_cpu.scalar_type() == torch::ScalarType::Float)
            {
                assert(batch_states.size() == static_cast<std::size_t>(heuristics_cpu.numel()));
                float* heuristics_cpu_data = heuristics_cpu.data_ptr<float>();

                for (int64_t batch_index = 0; batch_index < heuristics_cpu.numel(); ++batch_index)
                {
                    const auto costs_index = batch_indices[batch_index];
                    costs[costs_index] = static_cast<double>(heuristics_cpu_data[batch_index]);
                }
            }

            else if (heuristics_cpu.scalar_type() == torch::ScalarType::Double)
            {
                assert(batch_states.size() == static_cast<std::size_t>(heuristics_cpu.numel()));
                double* heuristics_cpu_data = heuristics_cpu.data_ptr<double>();

                for (int64_t batch_index = 0; batch_index < heuristics_cpu.numel(); ++batch_index)
                {
                    const auto costs_index = batch_indices[batch_index];
                    costs[costs_index] = heuristics_cpu_data[batch_index];
                }
            }
            else
            {
                throw std::runtime_error("Unsupported data type for heuristics_cpu");
            }

            for (std::size_t batch_index = 0; batch_index < batch_states.size(); ++batch_index)
            {
                const auto costs_index = batch_indices[batch_index];
                cache_.insert(std::make_pair(batch_states[batch_index], costs[costs_index]));
            }
        }

        return costs;
    }
}  // namespace planners
