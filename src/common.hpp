#ifndef COMMON_HPP
#define COMMON_HPP

#include <array>
#include <complex>
#include <forward_list>
#include <vector>
#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

using scalar_type = double;
// using data_type = std::complex<scalar_type>;
using data_type = thrust::complex<scalar_type>;
using cpudata_t = std::vector<scalar_type>;
using dim_t = std::array<size_t, 3>;


// const int threadsPerBlock{512};
extern int threadsPerBlock;


struct Variables {
    int KX, KY, KZ;
    int VX, VY, VZ;
    int BX, BY, BZ;
    int TH;
    int VEL;
    int MAG;

    int NUM_FIELDS;
    std::vector<std::string> VAR_LIST;
};

extern struct Variables vars;

struct Grid {
    size_t NX, NY, NZ;
    size_t FFT_SIZE[3];
    size_t NTOTAL, NTOTAL_COMPLEX;
};

extern struct Grid grid;


struct Parser {
    int nx, ny, nz;
    int stats_frequency;
    double max_hours;
    int restart_num;
    std::string input_dir;
    std::string output_dir;
};

#endif
