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

// __global__ void ComplexVecAssign(const cufftDoubleComplex *A, cufftDoubleComplex *B, size_t N) {
//     size_t i = i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
//
//     if (i < N) {
//         B[i].x = A[i].x;
//         B[i].y = A[i].y;
//     }
// }

__global__ void ComplexVecAssign(const data_type *A, data_type *B, size_t N) {
    size_t i = i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        B[i] = A[i];
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

__global__ void VecInit( scalar_type *X, scalar_type a, size_t N) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        X[i] = a;
    }
}

__global__ void VecInitComplex( data_type *X, data_type a, size_t N) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        X[i] = a;
    }
}

// equivalent of a*X + b*Y: I interpret complex vector as double, out-of-place (unless Z = X), a is double
__global__ void axpyDouble( scalar_type *X,  scalar_type *Y, scalar_type *Z, scalar_type a, scalar_type b, size_t N) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        Z[i] = a*X[i] + b*Y[i];
        // Y[i] = 0.0*Y[i];
    }
}

// equivalent of a*X + b*Y: I use complex vectors, out-of-place (unless Z = X), a is double
// __global__ void axpyComplex( const cufftDoubleComplex *X, const cufftDoubleComplex *Y, cufftDoubleComplex *Z, scalar_type a, scalar_type b, size_t N) {
//     size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
//
//     if (i < N) {
//         Z[i].x = a*X[i].x + b*Y[i].x;
//         Z[i].y = a*X[i].y + b*Y[i].y;
//     }
// }

__global__ void axpyComplex( const data_type *X, data_type *Y, data_type *Z, scalar_type a, scalar_type b, size_t N) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        Z[i] = a*X[i] + b*Y[i];
        // Y[i] = data_type(0.0,0.0);
    }
}

__global__ void axpy5ComplexAssign( data_type *A, data_type *B, data_type *C, data_type *D, data_type *E, scalar_type a, scalar_type b, scalar_type c, scalar_type d, scalar_type e, size_t N) {
    // real Y = mu_j*Uc(nv,k,j,i) + nu_j*Uc1(nv,k,j,i);
    // Uc1(nv,k,j,i) = Uc(nv,k,j,i);
    // Uc <- Y + (1.0 - mu_j - nu_j)*Uc0 + dt_hyp*mu_tilde_j*dU +  gamma_j*dt_hyp*dU0;
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    data_type Y = data_type(0.0,0.0);

    if (i < N) {
        Y = a*A[i] + b*B[i];
        B[i] = A[i];
        A[i] = Y + c*C[i] + d*D[i] + e*E[i];
    }
}

__global__ void addReset( const data_type *X, data_type *Y, data_type *Z, scalar_type a, scalar_type b, size_t N) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        Z[i] = a*X[i] + b*Y[i];
        Y[i] = data_type(0.0,0.0);
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

// __global__ void scalarDissipation( const scalar_type *d_all_kvec, const data_type *X, scalar_type *Z, size_t N) {
//     size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
//     // int vars.KX = 0; int vars.KY = 1; int vars.KZ = 2;
//     // this is the imaginary unit
//     // data_type imI = data_type(0.0,1.0);
//
//
//     if (i < N) {
//         Z[i] = - (d_all_kvec[vars.KX * N + i] * d_all_kvec[vars.KX * N + i] + d_all_kvec[vars.KY * N + i] * d_all_kvec[vars.KY * N + i] + d_all_kvec[vars.KZ * N + i] * d_all_kvec[vars.KZ * N + i] ) * (X[i].real() * X[i].real() + X[i].imag() * X[i].imag());
//     }
//
//
// }
