### these do not need to be modified
include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(
${PROJECT_SOURCE_DIR}/src/output/
)
include_directories(${CMAKE_CURRENT_LIST_DIR}/)
include(list_source_files.txt)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data/)

### these can be modified
add_executable(test_alfven-wave
# spooky.cu
initflow.cu
spooky.hpp
physics_modules.hpp
${spooky_src})


### do not modify

#target_link_libraries(test_taylor-green CUDA::cufft CUDA::cublas -lconfig -lhdf5)
#target_link_libraries(test_taylor-green CUDA::cufft CUDA::cublas CUDA::cudart HDF5::HDF5 ${LIBCONFIG})
target_link_libraries(test_alfven-wave CUDA::cufft CUDA::cublas CUDA::cudart hdf5-lib config)
target_link_libraries(test_alfven-wave HighFive)
target_link_libraries(test_alfven-wave argparse)



### can modify


add_test(
  NAME spooky-alfven-wave
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/check-alfven-wave.py --executable $<TARGET_FILE:test_alfven-wave> --input-dir ${CMAKE_CURRENT_LIST_DIR} --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  )

