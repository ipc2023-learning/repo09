#if !defined(CONVOLUTIONAL_SMOOTH_MAX_MODULE_HPP_)
#define CONVOLUTIONAL_SMOOTH_MAX_MODULE_HPP_

#include "torch/torch.h"

#include <vector>

class ConvolutionalSmoothMaximumModuleImpl : public torch::nn::Module
{
  private:
    double alpha_;
    torch::nn::Conv2d convolution_;

  public:
    ConvolutionalSmoothMaximumModuleImpl();

    ConvolutionalSmoothMaximumModuleImpl(int32_t in_features, int32_t out_features, double alpha);

    torch::Tensor forward(const torch::Tensor& input);
};

TORCH_MODULE(ConvolutionalSmoothMaximumModule);

#endif  // CONVOLUTIONAL_SMOOTH_MAX_MODULE_HPP_
