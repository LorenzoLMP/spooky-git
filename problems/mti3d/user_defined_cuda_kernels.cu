#include <cuda_runtime.h>
#include <cufftXt.h>
// #include "spooky.hpp"
#include "cufft_utils.h"
#include "define_types.hpp"
// #include "cufft_routines.hpp"
#include "common.hpp"


// #if defined(MHD) && defined(BOUSSINESQ) && defined(ANISOTROPIC_DIFFUSION)
// compute the vector ( b b_z) (in real space), and assign it to a new scratch 3D array
// B points at the memory location of the 3D vector, Z is the 3D scratch array
__global__ void Computebbstrat( const scalar_type *B,  scalar_type *Z, size_t N, int strat_dir) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    scalar_type B2 = 0.0;

    if (i < N) {
        B2 = B[ 0 * N + i ] * B[ 0 * N + i ] + B[ 1 * N + i ] * B[ 1 * N + i ] + B[ 2 * N + i ] * B[ 2 * N + i ] ;
        // Z_x = B_x B_stratdir / B^2
        Z[ 0 * N + i ] = B[ 0 * N + i ] *  B[ strat_dir * N + i ]  / B2 ;
        // Z_y = B_y B_stratdir / B^2
        Z[ 1 * N + i ] = B[ 1 * N + i ] *  B[ strat_dir * N + i ]  / B2 ;
        // Z_z = B_z B_stratdir / B^2
        Z[ 2 * N + i ] = B[ 2 * N + i ] *  B[ strat_dir * N + i ]  / B2 ;
    }
}

// #endif
