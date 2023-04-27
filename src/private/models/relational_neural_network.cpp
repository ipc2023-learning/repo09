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


#include "relational_neural_network.hpp"

namespace models
{
    RelationalNeuralNetwork::RelationalNeuralNetwork() : relational_neural_network_(nullptr) {}

    RelationalNeuralNetwork::RelationalNeuralNetwork(const RelationalMessagePassingNeuralNetwork& network) :
        relational_neural_network_(network),
        relational_transformer_(nullptr)
    {
    }

    RelationalNeuralNetwork::RelationalNeuralNetwork(const RelationalTransformer& network) :
        relational_neural_network_(nullptr),
        relational_transformer_(network)
    {
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RelationalNeuralNetwork::forward(const formalism::StateTransitions& state_transitions)
    {
        if (relational_neural_network_)
        {
            return relational_neural_network_->forward(state_transitions);
        }

        if (relational_transformer_)
        {
            return relational_transformer_->forward(state_transitions);
        }

        throw std::runtime_error("not implemented");
    }

    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
    RelationalNeuralNetwork::forward(const formalism::StateTransitionsVector& state_transitions)
    {
        if (relational_neural_network_)
        {
            return relational_neural_network_->forward(state_transitions);
        }

        if (relational_transformer_)
        {
            return relational_transformer_->forward(state_transitions);
        }

        throw std::runtime_error("not implemented");
    }

    std::pair<torch::Tensor, torch::Tensor> RelationalNeuralNetwork::forward(const formalism::StateProblemList& state_problems)
    {
        if (relational_neural_network_)
        {
            return relational_neural_network_->forward(state_problems);
        }

        if (relational_transformer_)
        {
            return relational_transformer_->forward(state_problems);
        }

        throw std::runtime_error("not implemented");
    }

    std::pair<torch::Tensor, torch::Tensor> RelationalNeuralNetwork::forward(const formalism::StateList& states, const formalism::ProblemDescription& problem)
    {
        if (relational_neural_network_)
        {
            return relational_neural_network_->forward(states, problem);
        }

        if (relational_transformer_)
        {
            return relational_transformer_->forward(states, problem);
        }

        throw std::runtime_error("not implemented");
    }

    std::pair<torch::Tensor, torch::Tensor>
    RelationalNeuralNetwork::forward(const formalism::StateList& states, const formalism::ProblemDescription& problem, uint32_t chunk_size)
    {
        if (relational_neural_network_)
        {
            return relational_neural_network_->forward(states, problem, chunk_size);
        }

        if (relational_transformer_)
        {
            return relational_transformer_->forward(states, problem, chunk_size);
        }

        throw std::runtime_error("not implemented");
    }

    void RelationalNeuralNetwork::to(c10::Device device, bool non_blocking)
    {
        if (relational_neural_network_)
        {
            relational_neural_network_->to(device, non_blocking);
        }

        if (relational_transformer_)
        {
            relational_transformer_->to(device, non_blocking);
        }
    }

    void RelationalNeuralNetwork::train(bool on)
    {
        if (relational_neural_network_)
        {
            relational_neural_network_->train(on);
        }

        if (relational_transformer_)
        {
            relational_transformer_->train(on);
        }
    }

    void RelationalNeuralNetwork::zero_grad(bool set_to_none)
    {
        if (relational_neural_network_)
        {
            relational_neural_network_->zero_grad(set_to_none);
        }

        if (relational_transformer_)
        {
            relational_transformer_->zero_grad(set_to_none);
        }
    }

    void RelationalNeuralNetwork::eval()
    {
        if (relational_neural_network_)
        {
            relational_neural_network_->eval();
        }

        if (relational_transformer_)
        {
            relational_transformer_->eval();
        }
    }

    torch::autograd::variable_list RelationalNeuralNetwork::parameters(bool recurse)
    {
        if (relational_neural_network_)
        {
            return relational_neural_network_->parameters();
        }

        if (relational_transformer_)
        {
            return relational_transformer_->parameters();
        }

        throw std::runtime_error("not implemented");
    }

    std::vector<std::pair<std::string, int32_t>> RelationalNeuralNetwork::predicates() const
    {
        if (relational_neural_network_)
        {
            return relational_neural_network_->predicates();
        }

        if (relational_transformer_)
        {
            return relational_transformer_->predicates();
        }

        throw std::runtime_error("not implemented");
    }

    DerivedPredicateList RelationalNeuralNetwork::derived_predicates() const
    {
        if (relational_neural_network_)
        {
            return relational_neural_network_->derived_predicates();
        }

        if (relational_transformer_)
        {
            return relational_transformer_->derived_predicates();
        }

        throw std::runtime_error("not implemented");
    }

    torch::Device RelationalNeuralNetwork::device() const
    {
        if (relational_neural_network_)
        {
            return relational_neural_network_->device();
        }

        if (relational_transformer_)
        {
            return relational_transformer_->device();
        }

        throw std::runtime_error("not implemented");
    };

    RelationalMessagePassingNeuralNetwork RelationalNeuralNetwork::get_relational_neural_network() const { return relational_neural_network_; }

    RelationalTransformer RelationalNeuralNetwork::get_relational_transformer() const { return relational_transformer_; }

    std::ostream& operator<<(std::ostream& os, const RelationalNeuralNetwork& network)
    {
        if (network.get_relational_neural_network())
        {
            os << network.get_relational_neural_network();
        }

        if (network.get_relational_transformer())
        {
            os << network.get_relational_transformer();
        }

        return os;
    }
}  // namespace models
