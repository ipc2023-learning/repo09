#if !defined(MODELS_MULTIHEAD_ATTENTION_MODULE_HPP_)
#define MODELS_MULTIHEAD_ATTENTION_MODULE_HPP_

#include "torch/torch.h"

namespace models
{
    class MultiheadAttentionImpl : public torch::nn::Module
    {
      private:
        int32_t input_size_;
        int32_t output_size_;
        int32_t num_heads_;
        torch::nn::Linear linear_query_;
        torch::nn::Linear linear_key_;
        torch::nn::Linear linear_value_;
        torch::nn::Linear linear_final_;

      public:
        MultiheadAttentionImpl(int32_t input_size, int32_t output_size, int32_t num_heads);

        torch::Tensor forward(const torch::Tensor& input);
    };

    TORCH_MODULE(MultiheadAttention);

}  // namespace models

#endif  // MODELS_MULTIHEAD_ATTENTION_MODULE_HPP_
