#if !defined(MODELS_UTILS_HPP_)
#define MODELS_UTILS_HPP_

#include "relational_neural_network.hpp"

// Older versions of LibC++ does not have filesystem (e.g., ubuntu 18.04), use the experimental version
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

namespace models
{
    void save_model(const fs::path& path, const models::RelationalNeuralNetwork& model);

    bool load_model(models::RelationalNeuralNetwork* loaded_model, const fs::path& path);

}  // namespace models

#endif  // MODELS_UTILS_HPP_
