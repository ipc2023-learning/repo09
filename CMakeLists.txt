cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
project(relnn)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  # Set up MSVC compiler
  list(APPEND CMAKE_PREFIX_PATH "D:/Libs/libtorch")
  file(GLOB TORCH_DLLS "D:/Libs/libtorch/lib/*.dll")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W1 /EHsc /bigobj /MP")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  set(CUDNN_INCLUDE_PATH "C:/Program Files/NVIDIA/CUDNN/v8.3/include")
  set(CUDNN_LIBRARY_PATH "C:/Program Files/NVIDIA/CUDNN/v8.3/lib/x64/cudnn.lib")
  set(BOOST_ROOT "C:/Libs/boost_1_81_0")
  include_directories("C:/Libs/boost_1_81_0")
elseif (CMAKE_COMPILER_IS_GNUCXX)
  # Enable relative position to LibTorch and the most common locations of CUDA
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "$ORIGIN/lib:/usr/local/libtorch/lib:/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
  # Set up GNU C++ compiler
  list(APPEND CMAKE_PREFIX_PATH "/usr/local/libtorch")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -DNDEBUG -flto -fwhole-program -funroll-loops -fomit-frame-pointer")
  set(CMAKE_CXX_FLAGS_PROFILING "-O3 -Wall -pg")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -g3 -ggdb")
endif()

find_package(Torch REQUIRED)
find_package(Boost REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})

set(default_build_type "Release")

add_subdirectory(src)
