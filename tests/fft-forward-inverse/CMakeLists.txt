### these do not need to be modified
include_directories(
${PROJECT_SOURCE_DIR}/src/
${PROJECT_SOURCE_DIR}/src/cuda/
${PROJECT_SOURCE_DIR}/src/fields/
${PROJECT_SOURCE_DIR}/src/io/
${PROJECT_SOURCE_DIR}/src/physics/
${PROJECT_SOURCE_DIR}/src/timestepping/
${PROJECT_SOURCE_DIR}/src/parameters/
${PROJECT_SOURCE_DIR}/src/supervisor/
${PROJECT_SOURCE_DIR}/src/rkl/
${PROJECT_SOURCE_DIR}/src/tests/
)


### do not modify
include_directories(${CMAKE_CURRENT_LIST_DIR}/)
include(list_source_files.txt)
add_executable(test_fft-forward-inverse
main.cu
cufft_routines.cu
${spooky_src})

target_link_libraries(test_fft-forward-inverse CUDA::cufft CUDA::cublas CUDA::cudart hdf5-lib config)
target_link_libraries(test_fft-forward-inverse HighFive)
target_link_libraries(test_fft-forward-inverse argparse)


### these can be modified
add_test(
  NAME spooky-fft-forward-inverse
  COMMAND test_fft-forward-inverse
)

