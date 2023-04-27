/*
 * Copyright (C) 2023 Simon Stahlberg
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */


#include "system.hpp"

#include <cstddef>
#include <iostream>

#if defined(_WIN32) || defined(_WIN64)
#include <psapi.h>
#include <windows.h>
#elif defined(__linux__)
#include <fstream>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace resources
{
    std::size_t get_memory_usage()
    {
        std::size_t memory_usage = 0;

#if defined(_WIN32) || defined(_WIN64)
        PROCESS_MEMORY_COUNTERS_EX pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*) &pmc, sizeof(pmc)))
        {
            memory_usage = pmc.PrivateUsage;
        }
#elif defined(__linux__)
        long rss = 0;
        FILE* fp = fopen("/proc/self/statm", "r");
        if (fp != nullptr)
        {
            if (fscanf(fp, "%*s%ld", &rss) != 1)
            {
                fclose(fp);
                return 0;  // Something went wrong, return 0
            }
            fclose(fp);
            memory_usage = rss * sysconf(_SC_PAGESIZE);
        }
#endif

        return memory_usage;
    }
}  // namespace resources
