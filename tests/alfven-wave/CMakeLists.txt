### these do not need to be modified
include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(
${PROJECT_SOURCE_DIR}/src/io/
)
include_directories(${CMAKE_CURRENT_LIST_DIR}/)
include(list_source_files.txt)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data/)

### these can be modified
add_executable(test_alfven-wave
initflow.cu
user_outputs.cu
${spooky_src})


### do not modify

target_link_libraries(test_alfven-wave CUDA::cufft CUDA::cublas CUDA::cudart hdf5-lib config)
target_link_libraries(test_alfven-wave HighFive)
target_link_libraries(test_alfven-wave argparse)



### can modify


add_test(
  NAME spooky-alfven-wave
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/check-alfven-wave.py --executable $<TARGET_FILE:test_alfven-wave> --input-dir ${CMAKE_CURRENT_LIST_DIR} --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  )

