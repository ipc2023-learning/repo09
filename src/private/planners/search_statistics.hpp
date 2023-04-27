#if !defined(PLANNERS_SEARCH_STATISTICS_HPP_)
#define PLANNERS_SEARCH_STATISTICS_HPP_

#include <chrono>

namespace planners
{
    struct SearchStatistics
    {
        uint32_t num_expanded;
        uint32_t num_generated;

        std::chrono::high_resolution_clock::time_point time_start;
        std::chrono::high_resolution_clock::time_point time_end;

        std::chrono::high_resolution_clock::duration duration_heuristic_seconds;
        std::chrono::high_resolution_clock::duration duration_successor_generator_seconds;

        inline void reset()
        {
            num_expanded = 0;
            num_generated = 0;

            time_start = std::chrono::high_resolution_clock::now();
            time_end = std::chrono::high_resolution_clock::now();

            duration_heuristic_seconds = std::chrono::seconds(0);
            duration_successor_generator_seconds = std::chrono::seconds(0);
        }
    };
}  // namespace planners

#endif  // PLANNERS_SEARCH_STATISTICS_HPP_
