#if !defined(RANDOM_DATASET_HPP_)
#define RANDOM_DATASET_HPP_

#include "../planners/state_space.hpp"
#include "dataset.hpp"

#include <random>
#include <vector>

namespace datasets
{
    class RandomDataset : public Dataset
    {
      private:
        planners::StateSpaceList state_spaces_;
        planners::StateSpaceSampleList samples_;
        bool random_sampling_;

      public:
        RandomDataset(const planners::StateSpaceList& state_spaces,
                      const bool random_sampling,
                      const bool remove_dead_ends = false,
                      const bool remove_goal_states = false,
                      const int32_t k = -1);

        uint32_t size() const override;

        planners::StateSpaceSample get(uint32_t index) const override;

        planners::StateSpaceSampleList get_range(uint32_t index, uint32_t count) const override;

        std::vector<std::pair<std::string, int32_t>> get_predicate_name_and_arities() const override;
    };
}  // namespace datasets

#endif  // RANDOM_DATASET_HPP_
