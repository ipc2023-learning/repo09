#if !defined(RESOURCES_HPP_)
#define RESOURCES_HPP_

#include <cstddef>

namespace resources
{
    std::size_t get_memory_usage();

    std::size_t get_unused_memory();
}  // namespace resources

#endif  // RESOURCES_HPP_
