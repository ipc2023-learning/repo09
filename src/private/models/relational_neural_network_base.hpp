#if !defined(RELATIONAL_NEURAL_NETWORK_BASE_HPP_)
#define RELATIONAL_NEURAL_NETWORK_BASE_HPP_

#include "../formalism/declarations.hpp"
#include "../formalism/problem.hpp"
#include "torch/torch.h"

#include <map>
#include <string>
#include <vector>

namespace models
{
    using PredicateArity = std::pair<std::string, int32_t>;
    using PredicateArityList = std::vector<PredicateArity>;

    using DerivedPredicateParams = std::pair<std::string, std::vector<std::string>>;
    using DerivedPredicateCase = std::vector<DerivedPredicateParams>;
    using DerivedPredicateCaseList = std::vector<DerivedPredicateCase>;
    using DerivedPredicate = std::pair<DerivedPredicateParams, DerivedPredicateCaseList>;
    using DerivedPredicateList = std::vector<DerivedPredicate>;

    // A list of (predicate_name, num_bound_variables, num_free_variables, list of cases) tuples
    using InternalDerivedPredicateList =
        std::vector<std::tuple<std::string, int32_t, int32_t, std::vector<std::vector<std::pair<std::string, std::vector<int32_t>>>>>>;

    class RelationalNeuralNetworkBase : public torch::nn::Module
    {
      private:
        std::map<std::string, int32_t> predicate_arities_;
        std::map<std::string, int32_t> predicate_ids_;
        std::vector<std::pair<int32_t, int32_t>> id_arities_;
        DerivedPredicateList external_derived_predicates_;
        InternalDerivedPredicateList internal_derived_predicates_;
        torch::Tensor dummy_;

      public:
        RelationalNeuralNetworkBase();

        RelationalNeuralNetworkBase(const PredicateArityList& predicates, const DerivedPredicateList& derived_predicates);

        virtual ~RelationalNeuralNetworkBase() = default;

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(const formalism::StateTransitions& state_transitions);

        /**
         * @brief Infer policy, values, dead-ends.
         *
         * @param state_transitions A batch of pairs, where each pair is a state and its successor states.
         * @return A tuple where: the first tensor contains probability distributions of transitions; the second tensor contains values of successors; and the
         * third contains dead-end logits of transitions.
         */
        std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
        forward(const formalism::StateTransitionsVector& state_transitions);

        std::pair<torch::Tensor, torch::Tensor> forward(const formalism::StateProblemList& state_problems);

        /**
         * @brief Infers values and dead-ends.
         *
         * @param states A list of states.
         * @param goal A list of goal atoms.
         * @param objects A list of objects.
         * @return Values of the successor states and logits of dead-end classification of successor states.
         */
        std::pair<torch::Tensor, torch::Tensor> forward(const formalism::StateList& states, const formalism::ProblemDescription& problem);

        std::pair<torch::Tensor, torch::Tensor> forward(const formalism::StateList& states, const formalism::ProblemDescription& problem, uint32_t chunk_size);

        // TODO: Move these into source file.

        std::vector<std::pair<std::string, int32_t>> predicates() const
        {
            return std::vector<std::pair<std::string, int32_t>>(predicate_arities_.begin(), predicate_arities_.end());
        }

        DerivedPredicateList derived_predicates() const { return external_derived_predicates_; }

        torch::Device device() const { return this->dummy_.device(); };

      protected:
        // TODO: Move these into source file.

        const std::map<std::string, int32_t>& predicate_arities() const { return predicate_arities_; }

        const std::map<std::string, int32_t>& predicate_ids() const { return predicate_ids_; }

        const std::vector<std::pair<int32_t, int32_t>>& id_arities() const { return id_arities_; }

        virtual torch::Tensor readout_dead_end(const torch::Tensor& object_embeddings, const std::vector<int64_t>& batch_sizes) = 0;

        virtual torch::Tensor readout_value(const torch::Tensor& object_embeddings, const std::vector<int64_t>& batch_sizes) = 0;

        virtual std::vector<torch::Tensor> readout_dead_end(const torch::Tensor& object_embeddings,
                                                            const std::vector<int64_t>& batch_sizes,
                                                            const std::vector<std::tuple<int32_t, int32_t, int32_t>>& batch_slices) = 0;

        virtual std::vector<torch::Tensor> readout_value(const torch::Tensor& object_embeddings,
                                                         const std::vector<int64_t>& batch_sizes,
                                                         const std::vector<std::tuple<int32_t, int32_t, int32_t>>& batch_slices) = 0;

        virtual std::vector<torch::Tensor> readout_policy(const torch::Tensor& object_embeddings,
                                                          const std::vector<int64_t>& batch_sizes,
                                                          const std::vector<std::tuple<int32_t, int32_t, int32_t>>& batch_slices) = 0;

      private:
        virtual torch::Tensor internal_forward(const std::map<std::string, std::vector<int64_t>>& batch_states, const std::vector<int64_t>& batch_sizes) = 0;
    };
}  // namespace models

#endif  // RELATIONAL_NEURAL_NETWORK_BASE_HPP_
