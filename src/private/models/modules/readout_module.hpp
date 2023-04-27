#if !defined(READOUT_MODULE_HPP_)
#define READOUT_MODULE_HPP_

#include "mlp_module.hpp"
#include "torch/torch.h"

class ReadoutModuleImpl : public torch::nn::Module
{
  private:
    MLPModule value_module_;
    torch::Tensor dummy_;

  public:
    ReadoutModuleImpl() : value_module_(nullptr), dummy_() {}

    ReadoutModuleImpl(const int32_t hidden_size, const int32_t out_size);

    torch::Tensor forward(const torch::Tensor& object_embeddings, const std::vector<int64_t>& batch_sizes);

  private:
    torch::Device device() { return dummy_.device(); };
};

TORCH_MODULE(ReadoutModule);

#endif  // READOUT_MODULE_HPP_
