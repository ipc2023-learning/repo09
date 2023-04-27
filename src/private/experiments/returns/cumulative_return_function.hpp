#if !defined(EXPERIMENTS_CUMULATIVE_RETURN_FUNCTION_HPP_)
#define EXPERIMENTS_CUMULATIVE_RETURN_FUNCTION_HPP_

#include "return_function.hpp"

#include <memory>

namespace experiments
{
    class CumulativeReturnFunctionImpl : public ReturnFunctionImpl
    {
      public:
        torch::Tensor get(int32_t step_index, int32_t num_steps, double discount, const Trajectory& trajectory, const torch::Device device) const;
    };

    using CumulativeReturnFunction = std::shared_ptr<CumulativeReturnFunctionImpl>;
}  // namespace experiments

#endif  // EXPERIMENTS_CUMULATIVE_RETURN_FUNCTION_HPP_
