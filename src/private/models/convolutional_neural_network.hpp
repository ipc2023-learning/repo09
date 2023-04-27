#if !defined(CONVOLUTIONAL_NEURAL_NETWORK_HPP_)
#define CONVOLUTIONAL_NEURAL_NETWORK_HPP_

#include "../formalism/atom.hpp"
#include "../formalism/declarations.hpp"
#include "../formalism/state.hpp"
#include "modules/convolutional_smooth_maximum_module.hpp"
#include "torch/torch.h"

#include <string>
#include <vector>

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
    class ConvolutionalNeuralNetworkImpl : public torch::nn::Module
    {
      private:
        torch::nn::Sequential initialization_;
        torch::nn::Sequential message_;
        torch::nn::Sequential update_;
        torch::nn::Sequential readout_;
        ConvolutionalSmoothMaximumModule aggregation_;
        // torch::nn::Conv2d aggregation_;
        uint32_t layers_;
        torch::Tensor dummy_;

      public:
        ConvolutionalNeuralNetworkImpl();

        ConvolutionalNeuralNetworkImpl(uint32_t height, uint32_t width, uint32_t colors, uint32_t layers, uint32_t channels);

        torch::Tensor forward(const torch::Tensor& image);

        torch::Device device() const { return this->dummy_.device(); };
    };

    TORCH_MODULE(ConvolutionalNeuralNetwork);
}  // namespace models

#endif  // CONVOLUTIONAL_NEURAL_NETWORK_HPP_
