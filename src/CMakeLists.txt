 
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

### do not modify
include_directories(${CMAKE_CURRENT_LIST_DIR}/)
include(list_source_files.txt)

# add_library(spooky-lib OBJECT ${spooky_src} )
# target_link_libraries(spooky-lib CUDA::cufft CUDA::cublas CUDA::cudart config hdf5-lib HighFive argparse)

add_executable(spooky
${spooky_src})
target_link_libraries(spooky CUDA::cufft CUDA::cublas CUDA::cudart config hdf5-lib HighFive argparse)
