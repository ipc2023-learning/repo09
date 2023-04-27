#if !defined(DEBUG_HPP_)
#define DEBUG_HPP_

#include "torch/torch.h"

#include <iostream>

namespace experiments
{
    void print_tensor(const torch::Tensor& tensor)
    {
        std::cout << "---" << std::endl;
        std::cout << tensor.sizes() << std::endl;
        torch::print(tensor);
        std::cout << std::endl;
    }
}  // namespace experiments

#endif  // DEBUG_HPP_
