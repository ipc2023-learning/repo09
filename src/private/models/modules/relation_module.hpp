#if !defined(RELATION_MODULE_HPP_)
#define RELATION_MODULE_HPP_

#include "mlp_module.hpp"
#include "torch/torch.h"

class RelationModuleImpl : public torch::nn::Module
{
  private:
    MLPModule relation_;
    int32_t arity_;
    int32_t hidden_size_;

  public:
    RelationModuleImpl(const int32_t arity, const int32_t hidden_size);
    torch::Tensor forward(const torch::Tensor& object_embeddings, const torch::Tensor& relation_values);
};

TORCH_MODULE_IMPL(RelationModule, RelationModuleImpl);

#endif  // RELATION_MODULE_HPP_
