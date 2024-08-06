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

__global__ void nablaOpScalar( const scalar_type *d_all_kvec, const data_type *X, data_type *Z, scalar_type a, size_t N, int flag);

__global__ void nablaOpVector( const scalar_type *d_all_kvec, const data_type *X, data_type *Z, scalar_type a, size_t N, int flag);

// __global__ void nablaOp( scalar_type *kx, scalar_type *ky, scalar_type *kz, cufftDoubleComplex *X, cufftDoubleComplex *Z, scalar_type a, size_t N, int flag);

__global__ void TracelessShearMatrix( const scalar_type *d_all_fields, scalar_type *d_all_tmparray, size_t N);

__global__ void EnergyFluxVector( const scalar_type *d_all_fields, scalar_type *d_all_tmparray, size_t N);

__global__ void NonLinHydroAdv(const scalar_type *d_all_kvec, const data_type *ShearMatrix, data_type *d_all_dfields, const scalar_type *d_mask, size_t N);

__global__ void NonLinBoussinesqAdv(const scalar_type *d_all_kvec, const data_type *EnergyFlux, data_type *d_all_dfields, const scalar_type *d_mask, size_t N);

__global__ void BoussinesqStrat( const data_type *d_all_fields, data_type *d_all_dfields, double BV_freq2, size_t N, int strat_dir);

__global__ void TracelessShearMatrixMHD( const scalar_type *d_all_fields, scalar_type *d_all_tmparray, size_t N);

__global__ void MagneticEmf( const scalar_type *d_all_fields, scalar_type *d_all_tmparray, size_t N);

__global__ void MagneticShear(const scalar_type *d_all_kvec, const data_type *MagEmf, data_type *d_all_dfields, const scalar_type *d_mask, size_t N);

__global__ void GradPseudoPressure(const scalar_type *d_all_kvec, data_type *d_all_dfields, size_t N);

__global__ void Gradient( const scalar_type *d_all_kvec, const data_type *X, data_type *Z, size_t N);

__global__ void Divergence( const scalar_type *d_all_kvec, const data_type *X, data_type *Z, size_t N);

// __global__ void CleanDivergence( const scalar_type *d_all_kvec, const cufftDoubleComplex *X, cufftDoubleComplex *Z, size_t N);
__global__ void CleanDivergence( const scalar_type *d_all_kvec, const data_type *X, data_type *Z, size_t N);

__global__ void ComputeAnisotropicHeatFlux( const scalar_type *B, const scalar_type *BGradTheta, scalar_type *Z, scalar_type OmegaT2, scalar_type chi, size_t N, int strat_dir);

// absolute<T> computes the absolute value of a number f(x) -> |x|
template <typename T>
struct absolute
{
    __host__ __device__
        T operator()(const T& x) const {
            return fabs(x);
        }
};


// absolute3<T> computes the absolute value of a tuple of 3 doubles f(x) -> |x|
typedef thrust::tuple<scalar_type,scalar_type,scalar_type> Tuple3;
template <typename T>
struct absolute3
{
    __host__ __device__
        Tuple3 operator()(const Tuple3 & t0) const {
            // Tuple3 temp;
            // thrust::get<0>(temp) = fabs(thrust::get<0>(t0));
            // thrust::get<1>(temp) = fabs(thrust::get<1>(t0));
            // thrust::get<2>(temp) = fabs(thrust::get<2>(t0));
            return thrust::make_tuple<scalar_type,scalar_type,scalar_type>(fabs(thrust::get<0>(t0)),fabs(thrust::get<1>(t0)),fabs(thrust::get<2>(t0)));
        }
};

template<typename T>
struct MaxAbs
{
    __host__ __device__
        Tuple3 operator()(const Tuple3& t0, const Tuple3& t1) {
            Tuple3 temp;
            thrust::get<0>(temp) = (thrust::get<0>(t0) < thrust::get<0>(t1)) ?                 thrust::get<0>(t1) : thrust::get<0>(t0);
            thrust::get<1>(temp) = (thrust::get<1>(t0) < thrust::get<1>(t1)) ?                thrust::get<1>(t1) : thrust::get<1>(t0);
            thrust::get<2>(temp) = (thrust::get<2>(t0) < thrust::get<2>(t1)) ?                thrust::get<2>(t1) : thrust::get<2>(t0);
            return temp;
            // return thrust::make_tuple(thrust::get<0>(t0) + thrust::get<0>(t1), thrust::get<1>(t0) + thrust::get<1>(t1), thrust::get<2>(t0) + thrust::get<2>(t1));
        }
};

