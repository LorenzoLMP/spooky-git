### these do not need to be modified
include_directories(${PROJECT_SOURCE_DIR}/src/)

include_directories(${CMAKE_CURRENT_LIST_DIR}/)
include(list_source_files.txt)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data/)

### these can be modified
add_executable(test_advected-mhd-vortex
# spooky.cu
initflow.cu
spooky.hpp
physics_modules.hpp
${spooky_src})


### do not modify

#target_link_libraries(test_taylor-green CUDA::cufft CUDA::cublas -lconfig -lhdf5)
#target_link_libraries(test_taylor-green CUDA::cufft CUDA::cublas CUDA::cudart HDF5::HDF5 ${LIBCONFIG})
target_link_libraries(test_advected-mhd-vortex CUDA::cufft CUDA::cublas CUDA::cudart hdf5-lib config)
target_link_libraries(test_advected-mhd-vortex HighFive)
target_link_libraries(test_advected-mhd-vortex argparse)



### can modify


add_test(
  NAME spooky-advected-mhd-vortex
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/check-advected-mhd-vortex.py --executable $<TARGET_FILE:test_advected-mhd-vortex> --input-dir ${CMAKE_CURRENT_LIST_DIR} --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  )

