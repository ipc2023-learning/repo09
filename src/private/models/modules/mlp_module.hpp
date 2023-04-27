#if !defined(MLP_MODULE_HPP_)
#define MLP_MODULE_HPP_

#include "torch/torch.h"

#include <vector>

class MLPModuleImpl : public torch::nn::Module
{
  private:
    torch::nn::Linear first_;
    torch::nn::Linear second_;
    torch::nn::Linear reshape_;

  public:
    MLPModuleImpl();

    MLPModuleImpl(int32_t in_features, int32_t out_features);

    torch::Tensor forward(const torch::Tensor& input);
};

TORCH_MODULE(MLPModule);

#endif  // MLP_MODULE_HPP_
