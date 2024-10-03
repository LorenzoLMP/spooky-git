### these do not need to be modified
include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(
${PROJECT_SOURCE_DIR}/src/io/
)
include_directories(${CMAKE_CURRENT_LIST_DIR}/)
include(list_source_files.txt)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data/)

### these can be modified
add_executable(test_mti-eigenmode
initflow.cu
user_defined_cuda_kernels.cu
user_outputs.cu
${spooky_src})


### do not modify

target_link_libraries(test_mti-eigenmode CUDA::cufft CUDA::cublas CUDA::cudart hdf5-lib config)
target_link_libraries(test_mti-eigenmode HighFive)
target_link_libraries(test_mti-eigenmode argparse)



### can modify


add_test(
  NAME spooky-mti-eigenmode
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/check-mti-eigenmode.py --executable $<TARGET_FILE:test_mti-eigenmode> --input-dir ${CMAKE_CURRENT_LIST_DIR} --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  )

