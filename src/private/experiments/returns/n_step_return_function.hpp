#if !defined(EXPERIMENTS_N_STEP_RETURN_FUNCTION_HPP_)
#define EXPERIMENTS_N_STEP_RETURN_FUNCTION_HPP_

#include "return_function.hpp"

#include <memory>

namespace experiments
{
    class NStepReturnFunctionImpl : public ReturnFunctionImpl
    {
      public:
        torch::Tensor get(int32_t step_index, int32_t num_steps, double discount, const Trajectory& trajectory, const torch::Device device) const;
    };

    using NStepReturnFunction = std::shared_ptr<NStepReturnFunctionImpl>;
}  // namespace experiments

#endif  // EXPERIMENTS_N_STEP_RETURN_FUNCTION_HPP_
