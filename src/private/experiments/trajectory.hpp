#if !defined(EXPERIMENTS_TRAJECTORY_HPP_)
#define EXPERIMENTS_TRAJECTORY_HPP_

#include "../formalism/domain.hpp"
#include "../formalism/problem.hpp"
#include "../formalism/state.hpp"
#include "../planners/state_space.hpp"
#include "torch/torch.h"

namespace experiments
{
    struct TrajectoryStep
    {
        const planners::StateSpace state_space;
        const formalism::State state;
        const formalism::State successor_state;
        const torch::Tensor transition_probability;
        const torch::Tensor state_value;
        const torch::Tensor successor_value;
        const double reward;

        TrajectoryStep(const planners::StateSpace& state_space,
                       const formalism::State& state,
                       const formalism::State& successor_state,
                       const torch::Tensor& transition_probability,
                       const torch::Tensor& state_value,
                       const torch::Tensor& successor_value,
                       const double reward);
    };

    using Trajectory = std::vector<TrajectoryStep>;
    using TrajectoryList = std::vector<experiments::Trajectory>;
}  // namespace experiments

#endif  // EXPERIMENTS_TRAJECTORY_HPP_
