#if !defined(EXPERIMENTS_LAMBDA_RETURN_FUNCTION_HPP_)
#define EXPERIMENTS_LAMBDA_RETURN_FUNCTION_HPP_

#include "return_function.hpp"

#include <memory>

namespace experiments
{
    class LambdaReturnFunctionImpl : public ReturnFunctionImpl
    {
      private:
        double lambda_;

      public:
        LambdaReturnFunctionImpl(double lambda);

        torch::Tensor get(int32_t step_index, int32_t num_steps, double discount, const Trajectory& trajectory, const torch::Device device) const;
    };

    using LambdaReturnFunction = std::shared_ptr<LambdaReturnFunctionImpl>;
}  // namespace experiments

#endif  // EXPERIMENTS_LAMBDA_RETURN_FUNCTION_HPP_
