#ifndef COMMON_HPP
#define COMMON_HPP

#include <array>
#include <complex>
#include <forward_list>
#include <vector>
#include <thrust/complex.h>

using scalar_type = double;
// using data_type = std::complex<scalar_type>;
using data_type = thrust::complex<scalar_type>;
using cpudata_t = std::vector<scalar_type>;
using dim_t = std::array<size_t, 3>;


// const int threadsPerBlock{512};
extern int threadsPerBlock;

#define SET 0
#define ADD 1

struct Variables {
    int KX, KY, KZ;
    int VX, VY, VZ;
    int BX, BY, BZ;
    int TH;
    int VEL;
    int MAG;

    int NUM_FIELDS;
};

extern Variables vars;

struct Grid {
    size_t NX, NY, NZ;
    size_t FFT_SIZE[3];
    size_t NTOTAL, NTOTAL_COMPLEX;
};

extern Grid grid;

#endif
