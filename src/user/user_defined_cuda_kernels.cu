#include <cuda_runtime.h>
#include <cufftXt.h>
#include "common.hpp"
#include "cufft_utils.h"
// #include "cufft_routines.hpp"



// #if defined(MHD) && defined(BOUSSINESQ) && defined(ANISOTROPIC_DIFFUSION)
// compute the vector ( b b_z) (in real space), and assign it to a new scratch 3D array
// B points at the memory location of the 3D vector, Z is the 3D scratch array


// #endif


__global__ void bUnitvector(const scalar_type *Xvec, scalar_type* unitVec, size_t N) {

    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    scalar_type norm = 1e-16; // for safety

    if (i < N) {
        norm += sqrt( Xvec[ 0 * N + i ] * Xvec[ 0 * N + i ] + Xvec[ 1 * N + i ] * Xvec[ 1 * N + i ] + Xvec[ 2 * N + i ] * Xvec[ 2 * N + i ] );
        
        // unitVec_x = Xvec_x / norm
        unitVec[ 0 * N + i ] = Xvec[ 0 * N + i ] / norm ;
        // unitVec_y = Xvec_y / norm
        unitVec[ 1 * N + i ] = Xvec[ 1 * N + i ] / norm ;
        // unitVec_z = Xvec_z / norm
        unitVec[ 2 * N + i ] = Xvec[ 2 * N + i ] / norm ;
    }

}

__global__ void AngleHorizPlane(const scalar_type *Xvec, scalar_type* angle, size_t N) {

    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    scalar_type norm = 1e-16; // for safety

    if (i < N) {
        norm += sqrt( Xvec[ 0 * N + i ] * Xvec[ 0 * N + i ] + Xvec[ 1 * N + i ] * Xvec[ 1 * N + i ] );
        
        // angle = acos ( abs ( Xvec_x ) / norm )
        angle[ i ] = acosf ( fabs ( Xvec[ 0 * N + i ] ) / norm ) ;
        
    }

}

__global__ void ComputebGradTheta( const scalar_type *B, const scalar_type *GradTheta, scalar_type *Z, size_t N) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    scalar_type Bnorm = 1e-16; // for safety

    if (i < N) {

        Bnorm += sqrt( B[ 0 * N + i ] * B[ 0 * N + i ] + B[ 1 * N + i ] * B[ 1 * N + i ] + B[ 2 * N + i ] * B[ 2 * N + i ] );

        // Z = (Bx gradx theta + By grady theta + Bz gradz theta)/Bnorm
        Z[i] = (B[ 0 * N + i ] * GradTheta[ 0 * N + i ]  +  B[ 1 * N + i ] * GradTheta[ 1 * N + i ] + B[ 2 * N + i ] * GradTheta[ 2 * N + i ]) / Bnorm ;
    }
}