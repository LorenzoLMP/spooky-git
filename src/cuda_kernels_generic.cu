#include <cuda_runtime.h>
#include <cufftXt.h>
// #include "spooky.hpp"
#include "cufft_utils.h"
#include "define_types.hpp"
// #include "cufft_routines.hpp"
// #include "common.hpp"

__global__ void scaleKernel(cufftDoubleComplex *ft, scalar_type scale, size_t N) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        ft[i].x *= scale;
        ft[i].y *= scale;
    }
}

__global__ void RRvectorMultiply(const scalar_type *A, const scalar_type *B, scalar_type *C, scalar_type a, size_t N) {
    size_t i = i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = a * A[i] * B[i];
    }
}

__global__ void ComplexVecAssign(const cufftDoubleComplex *A, cufftDoubleComplex *B, size_t N) {
    size_t i = i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        B[i].x = A[i].x;
        B[i].y = A[i].y;
    }
}

__global__ void RRvectorDivide(const scalar_type *A, const scalar_type *B, scalar_type *C, scalar_type a, size_t N) {
    size_t i = i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = a * A[i] / B[i];
    }
}

__global__ void ComplexNorm(const cufftDoubleComplex *A, scalar_type *B, size_t N) {
    size_t i = i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        B[i] = A[i].x * A[i].x + A[i].y * A[i].y;
    }
}

__global__ void DoubleAbsolute(const scalar_type *A, scalar_type *B, size_t N) {
    size_t i = i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        B[i] = fabs(A[i]);
    }
}

// __global__ void RCvectorMultiply(const scalar_type *A, const cufftDoubleComplex *X, const cufftDoubleComplex *Z, scalar_type a, size_t N){
//     size_t i = i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
//
//     if (i < N) {
//         C[i] = a * A[i] * B[i];
//     }
// }

// __global__ void RvectorReciprocal(scalar_type *A, size_t N) {
//     size_t i = i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
//
//     if (i < N) {
//         A[i] = 1. / A[i];
//     }
// }

// same but out of place
__global__ void RvectorReciprocal(const scalar_type *A, scalar_type *B, size_t N) {
    size_t i = i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        B[i] = 1. / A[i];
    }
}



// // equivalent of a*X + Y: I interpret complex vector as double, in-place (X is modified), a is double
// __global__ void axpyDouble( scalar_type *X, const scalar_type *Y, scalar_type a, size_t N) {
//     size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
//
//     if (i < N) {
//         X[i] = a*X[i] + Y[i];
//     }
// }

// equivalent of a*X + b*Y: I interpret complex vector as double, out-of-place (unless Z = X), a is double
__global__ void axpyDouble( scalar_type *X,  scalar_type *Y, scalar_type *Z, scalar_type a, scalar_type b, size_t N) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        Z[i] = a*X[i] + b*Y[i];
    }
}

// equivalent of a*X + b*Y: I use complex vectors, out-of-place (unless Z = X), a is double
__global__ void axpyComplex( const cufftDoubleComplex *X, const cufftDoubleComplex *Y, cufftDoubleComplex *Z, scalar_type a, scalar_type b, size_t N) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        Z[i].x = a*X[i].x + b*Y[i].x;
        Z[i].y = a*X[i].y + b*Y[i].y;
    }
}

// computes a * nabla X, where X is complex vector, out-of-place (unless Z = X), a is double
// __global__ void nablaOp( scalar_type *kx, scalar_type *ky, scalar_type *kz, cufftDoubleComplex *X, cufftDoubleComplex *Z, scalar_type a, size_t N, int flag) {
//     size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
//
//     if ( flag == 0 ){ // overwrite i-th element
//         if (i < N) {
//             Z[i].x = - a * (kx[i] * kx[i] + ky[i] * ky[i] + kz[i] * kz[i] ) * X[i].x;
//             Z[i].y = - a * (kx[i] * kx[i] + ky[i] * ky[i] + kz[i] * kz[i] ) * X[i].y;
//         }
//     }
//     else if ( flag == 1) { // accumulate to i-th element
//         if (i < N) {
//             Z[i].x += - a * (kx[i] * kx[i] + ky[i] * ky[i] + kz[i] * kz[i] ) * X[i].x;
//             Z[i].y += - a * (kx[i] * kx[i] + ky[i] * ky[i] + kz[i] * kz[i] ) * X[i].y;
//         }
//     }
//
// }

