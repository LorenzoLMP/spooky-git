#include <array>
#include <complex>
#include <forward_list>
#include <vector>
// #include "spooky.hpp"
#include "define_types.hpp"
// #include "parameters.hpp"
// extern Parameters param;
// #include "physics_modules.hpp"

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
    size_t nx, ny, nz;
    size_t fft_size[3];
    size_t ntotal, ntotal_complex;
};

extern Grid grid;

// extern Parameters param;

