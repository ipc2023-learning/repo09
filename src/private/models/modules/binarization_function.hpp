#if !defined(BINARIZATION_FUNCTION_HPP_)
#define BINARIZATION_FUNCTION_HPP_

#include "torch/torch.h"

class BinarizationFunction : public torch::autograd::Function<BinarizationFunction>
{
  public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor input);

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs);
};

#endif  // BINARIZATION_FUNCTION_HPP_
