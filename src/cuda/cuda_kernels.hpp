#include <cuda_runtime.h>
#include <cufftXt.h>
#include "cufft_utils.h"
#include "common.hpp"
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



// __global__ void nablaOp( scalar_type *kx, scalar_type *ky, scalar_type *kz, cufftDoubleComplex *X, cufftDoubleComplex *Z, scalar_type a, size_t N, int flag);

__global__ void TracelessShearMatrix( const scalar_type *VelField, scalar_type *ShearMatrix, size_t N);

__global__ void ShearMatrix( const scalar_type *VelField, scalar_type *ShearMatrix, size_t N);

__global__ void NonLinHydroAdv(const scalar_type *kvec, const data_type *ShearMatrix, data_type *dVelField, const scalar_type *mask, size_t N);

__global__ void NonLinAdvection(const scalar_type *kvec, const data_type *ShearMatrix, data_type *VecOutput, const scalar_type *mask, size_t N);

__global__ void GradPseudoPressure(const scalar_type *kvec, data_type *dVelField, size_t N);

__global__ void GradPseudoPressureShearing(const scalar_type *kvec, data_type *dVelField, data_type *Velx, double shear, size_t N);

__global__ void TracelessShearMatrixMHD( const scalar_type *VelField, const scalar_type *MagField, scalar_type *TShearMatrix, size_t N);

__global__ void MagneticEmf( const scalar_type *VelField, const scalar_type *MagField, scalar_type *Emf, size_t N);

__global__ void MagneticShear(const scalar_type *kvec, const data_type *MagEmf, data_type *dMagField, const scalar_type *mask, size_t N);

__global__ void EnergyFluxVector( const scalar_type *VelField, const scalar_type *Theta, scalar_type *EnFlux, size_t N);

__global__ void NonLinBoussinesqAdv(const scalar_type *kvec, const data_type *EnergyFlux, data_type *dTheta, const scalar_type *mask, size_t N);

__global__ void BoussinesqStrat( const data_type *VelField, const data_type *Theta, data_type *dVelField, data_type *dTheta, double BV_freq2, size_t N, int strat_dir);

__global__ void ComputeBGradTheta( const scalar_type *B, const scalar_type *GradTheta, scalar_type *Z, size_t N);

__global__ void ComputeAnisotropicHeatFlux( const scalar_type *B, const scalar_type *BGradTheta, scalar_type *Z, scalar_type OmegaT2, scalar_type chi, size_t N, int strat_dir);


__global__ void Computebbstrat( const scalar_type *B,  scalar_type *Z, size_t N, int strat_dir);


__global__ void ShearingFlow( const data_type *complex_Vecx, data_type *complex_dVecy,  double shear, size_t N);

__global__ void CoriolisForce( const data_type *complex_Vecx, const data_type *complex_Vecy, data_type *complex_dVecx, data_type *complex_dVecy, double omega, size_t N);

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

