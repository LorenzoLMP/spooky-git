#include <array>
#include <complex>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <cmath>
// #include "common.hpp"
#include <cufftXt.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/complex.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include "nvtx3/nvtx3.hpp"
// #include "physics_modules.hpp"


// #include <cuda_runtime.h>

// const double c_pi =

// const int threadsPerBlock{512};
// extern int dimGrid, dimBlock;

const  size_t nx = 512;        // Attribute (int variable)
const  size_t ny = 512;        // Attribute (int variable)
const  size_t nz = 512;        // Attribute (int variable)

const size_t fft_size[3] = {nx, ny, nz};
const size_t ntotal = fft_size[0] * fft_size[1] * fft_size[2];
const size_t ntotal_complex = fft_size[0] * fft_size[1] * ((fft_size[2] / 2) + 1);


/*
using scalar_type = double;
using data_type = std::complex<scalar_type>;
using cpudata_t = std::vector<scalar_type>;
using dim_t = std::array<size_t, 3>;

extern Parameters param;*/

