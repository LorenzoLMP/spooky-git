#include <array>
#include <complex>
#include <vector>
// #include "spooky.hpp"
// #include <cufftXt.h>
#include <thrust/complex.h>

using scalar_type = double;
// using data_type = std::complex<scalar_type>;
using data_type = thrust::complex<scalar_type>;
using cpudata_t = std::vector<scalar_type>;
using dim_t = std::array<size_t, 3>;
