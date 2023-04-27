#if !defined(HELPERS_HPP_)
#define HELPERS_HPP_

#include "private/formalism/declarations.hpp"
#include "private/models/relational_neural_network.hpp"
#include "private/planners/state_space.hpp"
#include "torch/torch.h"

#include <algorithm>

// Older versions of LibC++ does not have filesystem (e.g., ubuntu 18.04), use the experimental version
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

bool contains(const std::vector<std::string>& strings, const std::string& string);

torch::Device load_device(bool force_cpu);

models::RelationalNeuralNetwork load_model(const std::string& path,
                                           const std::string& type,
                                           const formalism::PredicateList& predicates,
                                           const models::DerivedPredicateList& derived_predicates,
                                           const formalism::TypeList& types,
                                           int32_t num_features,
                                           int32_t num_layers,
                                           bool global_readout,
                                           double maximum_smoothness);

models::RelationalNeuralNetwork load_model(const std::string& path,
                                           const std::string& type,
                                           const std::vector<std::pair<std::string, int32_t>>& predicates,
                                           const models::DerivedPredicateList& derived_predicates,
                                           int32_t num_features,
                                           int32_t num_layers,
                                           bool global_readout,
                                           double maximum_smoothness);

formalism::ProblemDescriptionList load_problems(const fs::path& path);

planners::StateSpaceList compute_state_spaces(const formalism::ProblemDescriptionList& problems,
                                              uint32_t max_size,
                                              bool use_weisfeiler_leman,
                                              bool& pruning_is_safe,
                                              bool& pruning_is_useful,
                                              int32_t timeout_s = -1,
                                              int32_t max_memory_mb = -1);

planners::StateSpaceList compute_state_spaces(const formalism::ProblemDescriptionList& problems,
                                              uint32_t max_size,
                                              bool use_weisfeiler_leman,
                                              int32_t timeout_s = -1,
                                              int32_t max_memory_mb = -1);

template<typename InputType, typename ReturnType>
void print_vector(std::vector<InputType> vector, std::function<ReturnType(InputType)> func, std::string delimiter)
{
    for (uint32_t index = 0; index < vector.size(); ++index)
    {
        std::cout << func(vector.at(index));

        if (index + 1 < vector.size())
        {
            std::cout << delimiter;
        }
    }
}

#endif  // HELPERS_HPP_
