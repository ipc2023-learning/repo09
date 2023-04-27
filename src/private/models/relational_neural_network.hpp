#if !defined(RELATIONAL_NEURAL_NETWORK_HPP_)
#define RELATIONAL_NEURAL_NETWORK_HPP_

#include "../formalism/declarations.hpp"
#include "../formalism/problem.hpp"
#include "relational_message_passing_neural_network.hpp"
#include "relational_transformer.hpp"
#include "torch/torch.h"

namespace models
{
    /// @brief This class is used instead of polymorphism since polymorphism cannot easily be achieved in LibTorch.
    class RelationalNeuralNetwork
    {
      private:
        RelationalMessagePassingNeuralNetwork relational_neural_network_;
        RelationalTransformer relational_transformer_;

      public:
        RelationalNeuralNetwork();

        RelationalNeuralNetwork(const RelationalMessagePassingNeuralNetwork& network);

        RelationalNeuralNetwork(const RelationalTransformer& network);

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(const formalism::StateTransitions& state_transitions);

        std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
        forward(const formalism::StateTransitionsVector& state_transitions);

        std::pair<torch::Tensor, torch::Tensor> forward(const formalism::StateProblemList& state_problems);

        std::pair<torch::Tensor, torch::Tensor> forward(const formalism::StateList& states, const formalism::ProblemDescription& problem);

        std::pair<torch::Tensor, torch::Tensor> forward(const formalism::StateList& states, const formalism::ProblemDescription& problem, uint32_t chunk_size);

        void to(c10::Device device, bool non_blocking = false);

        void train(bool on = true);

        void zero_grad(bool set_to_none = false);

        void eval();

        torch::autograd::variable_list parameters(bool recurse = true);

        std::vector<std::pair<std::string, int32_t>> predicates() const;

        DerivedPredicateList derived_predicates() const;

        torch::Device device() const;

        RelationalMessagePassingNeuralNetwork get_relational_neural_network() const;

        RelationalTransformer get_relational_transformer() const;
    };

    std::ostream& operator<<(std::ostream& os, const RelationalNeuralNetwork& network);
}  // namespace models

#endif  // RELATIONAL_NEURAL_NETWORK_HPP_
