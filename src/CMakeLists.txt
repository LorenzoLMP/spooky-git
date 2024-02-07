
include_directories(
${PROJECT_SOURCE_DIR}/libs/
)

include_directories(
${PROJECT_SOURCE_DIR}/libs/HighFive/include/
)

include_directories(
${PROJECT_SOURCE_DIR}/problems/generic/
)

# include_directories(
# ${PROJECT_SOURCE_DIR}/src/
# )

# add_library(spooky_lib
# spooky.hpp
# fields.cu
# fields.hpp
# wavevector.cu
# hydro_mhd_advance.cu
# compute_dfield.cu
# compute_dt.cu
# hdf5_io.cu
# hdf5_io.hpp
# cublas_routines.cu
# cublas_routines.hpp
# cuda_kernels.cu
# cuda_kernels.hpp
# cufft_routines.cu
# cufft_routines.hpp
# define_types.hpp
# initflow.cu
# parameters.hpp
# physics_modules.hpp
# read_parameters.cu
# tests.cu
# tests.hpp
# )

# add_executable(spooky
# spooky.cu)

# target_link_libraries(spooky spooky_lib)
# target_link_libraries(spooky_lib CUDA::cufft CUDA::cublas -lconfig -lhdf5)
# target_link_libraries(spooky_lib HighFive)
# add_executable(spooky
# spooky.cu
# spooky.hpp
# fields.cu
# fields.hpp
# wavevector.cu
# hydro_mhd_advance.cu
# compute_dfield.cu
# compute_dt.cu
# hdf5_io.cu
# hdf5_io.hpp
# cublas_routines.cu
# cublas_routines.hpp
# cuda_kernels.cu
# cuda_kernels.hpp
# cufft_routines.cu
# cufft_routines.hpp
# define_types.hpp
# initflow.cu
# parameters.hpp
# physics_modules.hpp
# read_parameters.cu
# tests.cu
# tests.hpp
# )



include(list_source_files.txt)
add_executable(spooky ${spooky_src})

target_link_libraries(spooky CUDA::cufft CUDA::cublas -lconfig -lhdf5)

add_library(HighFive INTERFACE)
target_include_directories(HighFive INTERFACE ${PROJECT_SOURCE_DIR}$/libs/HighFive/include)

target_link_libraries(spooky HighFive)
target_link_libraries(spooky argparse)

