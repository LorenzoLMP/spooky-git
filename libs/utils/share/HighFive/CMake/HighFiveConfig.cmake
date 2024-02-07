function(copy_interface_properties target source)
  foreach(prop
          INTERFACE_COMPILE_DEFINITIONS
          INTERFACE_COMPILE_FEATURES
          INTERFACE_COMPILE_OPTIONS
          INTERFACE_INCLUDE_DIRECTORIES
          INTERFACE_LINK_LIBRARIES
          INTERFACE_SOURCES
          INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
    set_property(TARGET ${target} APPEND PROPERTY ${prop} $<TARGET_PROPERTY:${source},${prop}>)
  endforeach()
endfunction()

if(TARGET HighFive)
    return()
endif()


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was HighFiveConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

# Get HighFive targets
include("${CMAKE_CURRENT_LIST_DIR}/HighFiveTargets.cmake")

# Recreate combined HighFive
add_library(HighFive INTERFACE IMPORTED)
set_property(TARGET HighFive APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS MPI_NO_CPPBIND)  # No c++ bindings

# Ensure we activate required C++ std
if(NOT DEFINED CMAKE_CXX_STANDARD)
  if(CMAKE_VERSION VERSION_LESS 3.8)
    message(WARNING "HighFive requires minimum C++11. (C++14 for XTensor) \
        You may need to set CMAKE_CXX_STANDARD in you project")
  else()
    # A client request for a higher std overrides this
    target_compile_features(HighFive INTERFACE cxx_std_11)
  endif()
endif()

# If the user sets this flag, all dependencies are preserved.
# Useful in central deployments where dependencies are not prepared later
set(HIGHFIVE_USE_INSTALL_DEPS OFF CACHE BOOL "Use original Highfive dependencies")
if(HIGHFIVE_USE_INSTALL_DEPS)
  # If enabled in the deploy config, request c++14
  if(OFF AND NOT CMAKE_VERSION VERSION_LESS 3.8)
    set_property(TARGET HighFive APPEND PROPERTY INTERFACE_COMPILE_FEATURES cxx_std_14)
  endif()
  message(STATUS "HIGHFIVE 2.8.0: Using original dependencies (HIGHFIVE_USE_INSTALL_DEPS=YES)")
  copy_interface_properties(HighFive HighFive_HighFive)
  return()
endif()

# When not using the pre-built dependencies, give user options
if(DEFINED HIGHFIVE_USE_BOOST)
  set(HIGHFIVE_USE_BOOST ${HIGHFIVE_USE_BOOST} CACHE BOOL "Enable Boost Support")
else()
  set(HIGHFIVE_USE_BOOST Off CACHE BOOL "Enable Boost Support")
endif()
set(HIGHFIVE_USE_EIGEN "${HIGHFIVE_USE_EIGEN}" CACHE BOOL "Enable Eigen testing")
set(HIGHFIVE_USE_XTENSOR "${HIGHFIVE_USE_XTENSOR}" CACHE BOOL "Enable xtensor testing")
set(HIGHFIVE_PARALLEL_HDF5 OFF CACHE BOOL "Enable Parallel HDF5 support")
option(HIGHFIVE_VERBOSE "Enable verbose logging" OFF)

if(HIGHFIVE_USE_XTENSOR AND NOT CMAKE_VERSION VERSION_LESS 3.8)
  set_property(TARGET HighFive APPEND PROPERTY INTERFACE_COMPILE_FEATURES cxx_std_14)
endif()

if(NOT HighFive_FIND_QUIETLY)
  message(STATUS "HIGHFIVE 2.8.0: (Re)Detecting Highfive dependencies (HIGHFIVE_USE_INSTALL_DEPS=NO)")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/HighFiveTargetDeps.cmake")
foreach(dependency HighFive_libheaders libdeps)
    copy_interface_properties(HighFive ${dependency})
endforeach()

check_required_components(HighFive)
