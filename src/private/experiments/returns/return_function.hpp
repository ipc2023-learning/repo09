#if !defined(EXPERIMENTS_RETURN_FUNCTION_HPP_)
#define EXPERIMENTS_RETURN_FUNCTION_HPP_

#include "../trajectory.hpp"

#include <memory>

namespace experiments
{
    class ReturnFunctionImpl
    {
      public:
        virtual torch::Tensor get(int32_t step_index, int32_t num_steps, double discount, const Trajectory& trajectory, const torch::Device device) const = 0;

        virtual ~ReturnFunctionImpl() {};
    };

    using ReturnFunction = std::shared_ptr<ReturnFunctionImpl>;
}  // namespace experiments

#endif  // EXPERIMENTS_RETURN_FUNCTION_HPP_
