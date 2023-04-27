#if !defined(DATASET_HPP_)
#define DATASET_HPP_

#include "../planners/state_space.hpp"

#include <vector>

namespace datasets
{
    class Dataset
    {
      public:
        virtual ~Dataset() {}

        virtual uint32_t size() const = 0;

        virtual planners::StateSpaceSample get(uint32_t index) const = 0;

        virtual planners::StateSpaceSampleList get_range(uint32_t index, uint32_t count) const = 0;

        virtual std::vector<std::pair<std::string, int32_t>> get_predicate_name_and_arities() const = 0;
    };
}  // namespace datasets

#endif  // DATASET_HPP_
