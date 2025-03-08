#include <cuda_runtime.h>
#include <cufftXt.h>
#include "cufft_utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/complex.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include "common.hpp"

__global__ void bUnitvector(const scalar_type *Xvec, scalar_type* unitVec, size_t N);

__global__ void AngleHorizPlane(const scalar_type *Xvec, scalar_type* angle, size_t N);

// __global__ void bGradTheta(const scalar_type *B, scalar_type* angle, size_t N);