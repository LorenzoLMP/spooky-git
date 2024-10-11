include_directories(
${PROJECT_SOURCE_DIR}/src/
${PROJECT_SOURCE_DIR}/src/cuda/
${PROJECT_SOURCE_DIR}/src/fields/
${PROJECT_SOURCE_DIR}/src/generic/
${PROJECT_SOURCE_DIR}/src/io/
${PROJECT_SOURCE_DIR}/src/physics/
${PROJECT_SOURCE_DIR}/src/timestepping/
${PROJECT_SOURCE_DIR}/src/parameters/
)

include(list_source_files.txt)

add_executable(spooky ${spooky_src})


#target_link_libraries(spooky CUDA::cufft CUDA::cublas -lconfig -lhdf5)
#target_link_libraries(spooky CUDA::cufft CUDA::cublas CUDA::cudart HDF5::HDF5 ${LIBCONFIG})
target_link_libraries(spooky CUDA::cufft CUDA::cublas CUDA::cudart config hdf5-lib)
#add_dependencies(spooky libconfig_download)
#target_link_libraries(spooky CUDA::cufft CUDA::cublas CUDA::cudart libconfig++ hdf5-lib)

add_library(HighFive INTERFACE)
target_include_directories(HighFive INTERFACE ${PROJECT_SOURCE_DIR}$/libs/HighFive/include)

target_link_libraries(spooky HighFive)
target_link_libraries(spooky argparse)

