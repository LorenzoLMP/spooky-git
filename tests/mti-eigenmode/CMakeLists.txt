### these do not need to be modified
include_directories(${PROJECT_SOURCE_DIR}/src/)

include_directories(${CMAKE_CURRENT_LIST_DIR}/)
include(list_source_files.txt)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data/)

### these can be modified
add_executable(test_mti-eigenmode
# spooky.cu
initflow.cu
spooky.hpp
physics_modules.hpp
${spooky_src})


### do not modify

#target_link_libraries(test_taylor-green CUDA::cufft CUDA::cublas -lconfig -lhdf5)
#target_link_libraries(test_taylor-green CUDA::cufft CUDA::cublas CUDA::cudart HDF5::HDF5 ${LIBCONFIG})
target_link_libraries(test_mti-eigenmode CUDA::cufft CUDA::cublas CUDA::cudart hdf5-lib config)
target_link_libraries(test_mti-eigenmode HighFive)
target_link_libraries(test_mti-eigenmode argparse)



### can modify


add_test(
  NAME spooky-mti-eigenmode
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/check-mti-eigenmode.py --executable $<TARGET_FILE:test_mti-eigenmode> --input-dir ${CMAKE_CURRENT_LIST_DIR} --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  )

