
cmake_minimum_required(VERSION 3.24)

#if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#  set(CMAKE_CUDA_ARCHITECTURES 75)
#endif()

if(NOT DEFINED CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER /opt/nvidia/hpc_sdk/Linux_x86_64/2023/cuda/12.0/bin/nvcc)
endif()

# if(NOT DEFINED CUDAToolkit_ROOT)
#   set(CUDAToolkit_ROOT /opt/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/)
# endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_FLAGS)
  set(CMAKE_CUDA_FLAGS "-ccbin /usr/bin/g++-12")
endif()

# if(NOT DEFINED BUILD_TESTS)
#   set(BUILD_TESTS "False")
# endif()
option(BUILD_TESTS "Build test programs" OFF)

set(CMAKE_CXX_FLAGS "-O3 -std=c++17")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

project(spooky-git VERSION 0.1 LANGUAGES CUDA CXX)


find_package(CUDAToolkit)

find_package(Python REQUIRED)

include(CTest)

include(CheckLanguage)
check_language(CUDA)


include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

# set (CMAKE_RUNTIME_OUTPUT_DIRECTORY /home/lorenzolmp/Documents/AIP-postdoc/cuda-code/spooky-cmake/)

# set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ../../)
#set (CMAKE_BINARY_DIR build)



add_subdirectory(src)

if( BUILD_TESTS )
  enable_testing()
  add_subdirectory(tests)
endif( BUILD_TESTS )



