#include <cuda_runtime.h>
#include <cufftXt.h>
// #include "spooky.hpp"
#include "cufft_utils.h"
#include "define_types.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/complex.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>

// #include "cufft_routines.hpp"


// extern const int threadsPerBlock;
// extern int blocksPerGrid;

__global__ void scaleKernel(cufftDoubleComplex *ft, scalar_type scale, size_t N);
__global__ void RRvectorMultiply(const scalar_type *A, const scalar_type *B, scalar_type *C, scalar_type a, size_t N);
__global__ void RRvectorDivide(const scalar_type *A, const scalar_type *B, scalar_type *C, scalar_type a, size_t N);

// __global__ void RCvectorMultiply(const scalar_type *A, const cufftDoubleComplex *X, const cufftDoubleComplex *Z, scalar_type a, size_t N);

// __global__ void RvectorReciprocal(scalar_type *A, size_t N);
__global__ void RvectorReciprocal(const scalar_type *A, scalar_type *B, size_t N);
__global__ void axpyDouble( scalar_type *X,  scalar_type *Y, scalar_type *Z, scalar_type a, scalar_type b, size_t N);
__global__ void axpyComplex( const cufftDoubleComplex *X, const cufftDoubleComplex *Y, cufftDoubleComplex *Z, scalar_type a, scalar_type b, size_t N);

__global__ void ComplexVecAssign(const cufftDoubleComplex *A, cufftDoubleComplex *B, size_t N);

// __global__ void scalarDissipation( const scalar_type *d_all_kvec, const data_type *X, scalar_type *Z, size_t N);
