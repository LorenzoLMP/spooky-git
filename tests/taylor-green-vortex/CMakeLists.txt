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
${PROJECT_SOURCE_DIR}/src/user/
)

include_directories(${CMAKE_CURRENT_LIST_DIR}/)

include(list_source_files.txt)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data/)

### these can be modified
add_executable(test_taylor-green
initflow.cu
${spooky_src})


### do not modify

target_link_libraries(test_taylor-green CUDA::cufft CUDA::cublas CUDA::cudart hdf5-lib config)
target_link_libraries(test_taylor-green HighFive)
target_link_libraries(test_taylor-green argparse)



### can modify


add_test(
  NAME spooky-taylor-green
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/check-taylor-green.py --executable $<TARGET_FILE:test_taylor-green> --input-dir ${CMAKE_CURRENT_LIST_DIR} --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  )

