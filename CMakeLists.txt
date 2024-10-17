
cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()


if(NOT DEFINED CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER /opt/nvidia/hpc_sdk/Linux_x86_64/2023/cuda/12.0/bin/nvcc)
endif()

# if(NOT DEFINED CUDAToolkit_ROOT)
#   set(CUDAToolkit_ROOT /opt/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/)
# endif()

# if(NOT DEFINED CMAKE_CUDA_STANDARD)
#    set(CMAKE_CUDA_STANDARD 11)
#    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# endif()

if(NOT DEFINED CMAKE_CUDA_FLAGS)
  set(CMAKE_CUDA_FLAGS "-ccbin /usr/bin/g++-12")
endif()

option(BUILD_TESTS "Build test programs" OFF)

set(CMAKE_CXX_FLAGS "-O3 -std=c++17")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

project(spooky-git VERSION 0.1 LANGUAGES CUDA CXX C)
include(CMakePrintHelpers)
include(FetchContent)
find_package(CUDAToolkit REQUIRED)

find_package(Python REQUIRED)

cmake_print_variables(LIBCONFIG_ROOT)
find_library(libconfig "config" 
	HINTS "${LIBCONFIG_ROOT}/lib") 
cmake_print_variables(libconfig)
if(NOT libconfig)
	message("config library not found")
	FetchContent_Declare(libconfig
	    GIT_REPOSITORY https://github.com/hyperrealm/libconfig.git
	)
	FetchContent_MakeAvailable(libconfig)
	cmake_print_variables(libconfig_SOURCE_DIR)
	cmake_print_variables(libconfig_BINARY_DIR)
	include_directories(${libconfig_SOURCE_DIR} ${libconfig_SOURCE_DIR}/lib)
else()
	message("LIBCONFIG: found")
	include_directories(${LIBCONFIG_ROOT}/include)
	add_library(config SHARED IMPORTED)
	set_target_properties(config PROPERTIES IMPORTED_LOCATION ${libconfig})
endif()
# find_package(libconfig) 
# cmake_print_variables(libconfig_FOUND)
# if(NOT ${libconfig_FOUND})
# 	message("config library not found")
# 	FetchContent_Declare(libconfig
# 	    GIT_REPOSITORY https://github.com/hyperrealm/libconfig.git
# 	)
# 	FetchContent_MakeAvailable(libconfig)
# 	cmake_print_variables(libconfig_SOURCE_DIR)
# 	cmake_print_variables(libconfig_BINARY_DIR)
# 	include_directories(${libconfig_SOURCE_DIR} ${libconfig_SOURCE_DIR}/lib)
# else()
# 	message("LIBCONFIG: found")
# 	include_directories(${libconfig_ROOT}/include ${libconfig_ROOT}/lib)
# endif()
#cmake_print_variables(${libconfig})
#get_property(libconfig_SOURCE_DIR CACHE libconfig PROPERTY INCLUDE_DIRECTORIES)
#cmake_print_variables(libconfig_INCLUDE_DIRS)
#cmake_print_variables(LIBCONFIG_INCLUDE_DIRS)


find_package(HDF5)
if(NOT ${HDF5_FOUND})
	FetchContent_Declare(HDF5                              # name of the content
  	GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git    # the repository
  	GIT_TAG        hdf5-1_14_3                                 # the tag
  	)
	FetchContent_MakeAvailable(HDF5)
	#FetchContent_SetPopulated(HDF5)
	# FetchContent_GetProperties(HDF5)
	# set (HDF5_BUILD_CPP_LIB ON CACHE BOOL "Build C++ support" FORCE)
	# target_compile_options(HDF5 PUBLIC --enable-cxx)
	# if(NOT HDF5_POPULATED)
	# 	FetchContent_Populate(HDF5)
	# 	set (HDF5_BUILD_CPP_LIB ON CACHE BOOL "Build C++ support" FORCE)
	# 	add_subdirectory(${HDF5_SOURCE_DIR} ${HDF5_BINARY_DIR})
	# endif()
	add_library(hdf5-lib ALIAS hdf5-static)
	cmake_print_variables(HDF5_SOURCE_DIR)
	cmake_print_variables(HDF5_BINARY_DIR)
	include_directories(${HDF5_SOURCE_DIR} ${HDF5_SOURCE_DIR}/src ${HDF5_BINARY_DIR} ${HDF5_BINARY_DIR}/src)
else()
	add_library(hdf5-lib ALIAS HDF5::HDF5)
endif()

add_library(HighFive INTERFACE)
target_include_directories(HighFive INTERFACE ${PROJECT_SOURCE_DIR}$/libs/HighFive/include)

include(CTest)

include(CheckLanguage)
check_language(CUDA)


FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

include_directories(
${PROJECT_SOURCE_DIR}/libs/
)

include_directories(
${PROJECT_SOURCE_DIR}/libs/HighFive/include/
)

#include_directories(
#/home/lperrone/mylibconfig/libconfig/include/
#)


#add_subdirectory(src)
add_subdirectory(problems)


if( BUILD_TESTS )
  enable_testing()
  add_subdirectory(tests)
endif( BUILD_TESTS )



