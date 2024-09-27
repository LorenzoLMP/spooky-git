### these do not need to be modified
include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(
${PROJECT_SOURCE_DIR}/src/output/
)
include_directories(${CMAKE_CURRENT_LIST_DIR}/)
include(list_source_files.txt)

### these can be modified
add_executable(test_fft-forward-inverse
spooky.cu
${spooky_src})


### change name

target_link_libraries(test_fft-forward-inverse CUDA::cufft CUDA::cublas CUDA::cudart hdf5-lib config)
target_link_libraries(test_fft-forward-inverse HighFive)
target_link_libraries(test_fft-forward-inverse argparse)



### can modify

add_test(
  NAME spooky-fft-forward-inverse
  COMMAND test_fft-forward-inverse
)

