#if !defined(WALKER_DATASET_HPP_)
#define WALKER_DATASET_HPP_

#include "../planners/state_space.hpp"
#include "dataset.hpp"

#include <random>
#include <unordered_map>
#include <vector>

namespace datasets
{
    class FixedSizeBalancedDataset : public Dataset
    {
      private:
        planners::StateSpaceList state_spaces_;
        std::vector<std::unordered_map<uint32_t, std::vector<formalism::State>>> states_grouped_by_label_;
        mutable std::default_random_engine generator_;
        uint32_t size_;

      public:
        FixedSizeBalancedDataset(const planners::StateSpaceList& state_spaces, uint32_t size);

        uint32_t size() const override;

        planners::StateSpaceSample get(uint32_t index) const override;

        planners::StateSpaceSampleList get_range(uint32_t index, uint32_t count) const override;

        std::vector<std::pair<std::string, int32_t>> get_predicate_name_and_arities() const override;
    };
}  // namespace datasets

#endif  // WALKER_DATASET_HPP_
