#include <cuda_runtime.h>
#include <cufftXt.h>
// #include "spooky.hpp"
#include "cufft_utils.h"
#include "define_types.hpp"
// #include "cufft_routines.hpp"
#include "common.hpp"


__global__ void nablaOpScalar( const scalar_type *d_all_kvec, const cufftDoubleComplex *X, cufftDoubleComplex *Z, scalar_type a, size_t N, int flag) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // int KX = 0; int KY = 1; int KZ = 2;

    if ( flag == 0 ){ // overwrite i-th element
        if (i < N) {
            Z[i].x = - a * (d_all_kvec[KX * N + i] * d_all_kvec[KX * N + i] + d_all_kvec[KY * N + i] * d_all_kvec[KY * N + i] + d_all_kvec[KZ * N + i] * d_all_kvec[KZ * N + i] ) * X[i].x;
            Z[i].y = - a * (d_all_kvec[KX * N + i] * d_all_kvec[KX * N + i] + d_all_kvec[KY * N + i] * d_all_kvec[KY * N + i] + d_all_kvec[KZ * N + i] * d_all_kvec[KZ * N + i] ) * X[i].y;
        }
    }
    else if ( flag == 1) { // accumulate to i-th element
        if (i < N) {
            Z[i].x += - a * (d_all_kvec[KX * N + i] * d_all_kvec[KX * N + i] + d_all_kvec[KY * N + i] * d_all_kvec[KY * N + i] + d_all_kvec[KZ * N + i] * d_all_kvec[KZ * N + i] ) * X[i].x;
            Z[i].y += - a * (d_all_kvec[KX * N + i] * d_all_kvec[KX * N + i] + d_all_kvec[KY * N + i] * d_all_kvec[KY * N + i] + d_all_kvec[KZ * N + i] * d_all_kvec[KZ * N + i] ) * X[i].y;
        }
    }

}


__global__ void nablaOpVector( const scalar_type *d_all_kvec, const cufftDoubleComplex *X, cufftDoubleComplex *Z, scalar_type a, size_t N, int flag) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // int KX = 0; int KY = 1; int KZ = 2;
    // assuming that X and Z are pointing at the first element of the 3D vector, then this kernel also works for magnetic field

    if ( flag == 0 ){ // overwrite i-th element
        if (i < N) {
            // VX/BX component
            Z[i].x = - a * (d_all_kvec[KX * N + i] * d_all_kvec[KX * N + i] + d_all_kvec[KY * N + i] * d_all_kvec[KY * N + i] + d_all_kvec[KZ * N + i] * d_all_kvec[KZ * N + i] ) * X[i].x;
            Z[i].y = - a * (d_all_kvec[KX * N + i] * d_all_kvec[KX * N + i] + d_all_kvec[KY * N + i] * d_all_kvec[KY * N + i] + d_all_kvec[KZ * N + i] * d_all_kvec[KZ * N + i] ) * X[i].y;

            // VY/BY component
            Z[N + i].x = - a * (d_all_kvec[KX * N + i] * d_all_kvec[KX * N + i] + d_all_kvec[KY * N + i] * d_all_kvec[KY * N + i] + d_all_kvec[KZ * N + i] * d_all_kvec[KZ * N + i] ) * X[N + i].x;
            Z[N + i].y = - a * (d_all_kvec[KX * N + i] * d_all_kvec[KX * N + i] + d_all_kvec[KY * N + i] * d_all_kvec[KY * N + i] + d_all_kvec[KZ * N + i] * d_all_kvec[KZ * N + i] ) * X[N + i].y;

            // VZ/BZ component
            Z[2 * N + i].x = - a * (d_all_kvec[KX * N + i] * d_all_kvec[KX * N + i] + d_all_kvec[KY * N + i] * d_all_kvec[KY * N + i] + d_all_kvec[KZ * N + i] * d_all_kvec[KZ * N + i] ) * X[2 * N + i].x;
            Z[2 * N + i].y = - a * (d_all_kvec[KX * N + i] * d_all_kvec[KX * N + i] + d_all_kvec[KY * N + i] * d_all_kvec[KY * N + i] + d_all_kvec[KZ * N + i] * d_all_kvec[KZ * N + i] ) * X[2 * N + i].y;
        }
    }
    else if ( flag == 1) { // accumulate to i-th element
        if (i < N) {
            // VX/BX component
            Z[i].x += - a * (d_all_kvec[KX * N + i] * d_all_kvec[KX * N + i] + d_all_kvec[KY * N + i] * d_all_kvec[KY * N + i] + d_all_kvec[KZ * N + i] * d_all_kvec[KZ * N + i] ) * X[i].x;
            Z[i].y += - a * (d_all_kvec[KX * N + i] * d_all_kvec[KX * N + i] + d_all_kvec[KY * N + i] * d_all_kvec[KY * N + i] + d_all_kvec[KZ * N + i] * d_all_kvec[KZ * N + i] ) * X[i].y;

            // VY/BY component
            Z[N + i].x += - a * (d_all_kvec[KX * N + i] * d_all_kvec[KX * N + i] + d_all_kvec[KY * N + i] * d_all_kvec[KY * N + i] + d_all_kvec[KZ * N + i] * d_all_kvec[KZ * N + i] ) * X[N + i].x;
            Z[N + i].y += - a * (d_all_kvec[KX * N + i] * d_all_kvec[KX * N + i] + d_all_kvec[KY * N + i] * d_all_kvec[KY * N + i] + d_all_kvec[KZ * N + i] * d_all_kvec[KZ * N + i] ) * X[N + i].y;

            // VZ/BZ component
            Z[2 * N + i].x += - a * (d_all_kvec[KX * N + i] * d_all_kvec[KX * N + i] + d_all_kvec[KY * N + i] * d_all_kvec[KY * N + i] + d_all_kvec[KZ * N + i] * d_all_kvec[KZ * N + i] ) * X[2 * N + i].x;
            Z[2 * N + i].y += - a * (d_all_kvec[KX * N + i] * d_all_kvec[KX * N + i] + d_all_kvec[KY * N + i] * d_all_kvec[KY * N + i] + d_all_kvec[KZ * N + i] * d_all_kvec[KZ * N + i] ) * X[2 * N + i].y;
        }
    }

}


#ifdef INCOMPRESSIBLE
// compute the elements of the traceless symmetric matrix B_ij = u_i u_j - delta_ij Tr (u_i u_j) / 3. It has only 5 independent components B_xx, B_xy, B_xz, Byy, B_yz. (B_zz = - B_xx - B_yy)
// the results are saved in the first 5 temp_arrays (after those reserved for the fields, the memory block points already at the right location)
__global__ void TracelessShearMatrix( const scalar_type *d_all_fields, scalar_type *d_all_tmparray, size_t N) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // int VX = 0; int VY = 1; int VZ = 2;
    if (i < N) {

        // 0: B_xx = u_x^2 - u^2/3
        d_all_tmparray[i] = ( 2.0 * d_all_fields[ VX * N + i] * d_all_fields[ VX * N + i] - d_all_fields[ VY * N + i] * d_all_fields[ VY * N + i] - d_all_fields[ VZ * N + i] * d_all_fields[ VZ * N + i] ) / 3.0;
        // 1: B_xy = u_x u_y
        d_all_tmparray[N + i] = d_all_fields[ VX * N + i] * d_all_fields[ VY * N + i] ;
        // 2: B_xz = u_x u_z
        d_all_tmparray[ 2 * N + i] = d_all_fields[ VX * N + i] * d_all_fields[ VZ * N + i] ;
        // 3: B_yy = u_y^2 - u^2/3
        d_all_tmparray[ 3 * N + i] = ( - d_all_fields[ VX * N + i] * d_all_fields[ VX * N + i] + 2.0 * d_all_fields[ VY * N + i] * d_all_fields[ VY * N + i] - d_all_fields[ VZ * N + i] * d_all_fields[ VZ * N + i] ) / 3.0;
        // 4: B_yz = u_y u_z
        d_all_tmparray[ 4 * N + i] = d_all_fields[ VY * N + i] * d_all_fields[ VZ * N + i] ;
    }
}



// compute derivative of traceless shear matrix and assign to dfields
__global__ void NonLinHydroAdv(const scalar_type *d_all_kvec, const cufftDoubleComplex *ShearMatrix, cufftDoubleComplex *d_all_dfields, const scalar_type *d_mask, size_t N){
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // int KX = 0; int KY = 1; int KZ = 2;
    // int VX = 0; int VY = 1; int VZ = 2;
    if (i < N) {
        // delta u_x = - ( I k_x B_xx + I k_y B_xy + I k_z B_xz)
        // operations divided in real and complex part
        d_all_dfields[VX * N + i].x =   d_mask[i] * (  d_all_kvec[KX * N + i] * ShearMatrix[i].y + d_all_kvec[KY * N + i] * ShearMatrix[N + i].y + d_all_kvec[KZ * N + i] * ShearMatrix[2 * N + i].y );
        d_all_dfields[VX * N + i].y =   d_mask[i] * (- d_all_kvec[KX * N + i] * ShearMatrix[i].x - d_all_kvec[KY * N + i] * ShearMatrix[N + i].x - d_all_kvec[KZ * N + i] * ShearMatrix[2 * N + i].x );

        // delta u_y = - ( I k_x B_yx + I k_y B_yy + I k_z B_yz)
        // operations divided in real and complex part
        d_all_dfields[VY * N + i].x =   d_mask[i] * (  d_all_kvec[KX * N + i] * ShearMatrix[N + i].y + d_all_kvec[KY * N + i] * ShearMatrix[3 * N + i].y + d_all_kvec[KZ * N + i] * ShearMatrix[4 * N + i].y );
        d_all_dfields[VY * N + i].y =   d_mask[i] * (- d_all_kvec[KX * N + i] * ShearMatrix[N + i].x - d_all_kvec[KY * N + i] * ShearMatrix[3 * N + i].x - d_all_kvec[KZ * N + i] * ShearMatrix[4 * N + i].x );

        // delta u_z = - ( I k_x B_zx + I k_y B_zy + I k_z B_zz)
        // note that B_zz = - B_xx - B_yy
        // operations divided in real and complex part
        d_all_dfields[VZ * N + i].x =   d_mask[i] * (  d_all_kvec[KX * N + i] * ShearMatrix[2 * N + i].y + d_all_kvec[KY * N + i] * ShearMatrix[4 * N + i].y - d_all_kvec[KZ * N + i] * ( ShearMatrix[i].y + ShearMatrix[3 * N + i].y ) );
        d_all_dfields[VZ * N + i].y =   d_mask[i] * (- d_all_kvec[KX * N + i] * ShearMatrix[2 * N + i].x - d_all_kvec[KY * N + i] * ShearMatrix[4 * N + i].x + d_all_kvec[KZ * N + i] * ( ShearMatrix[i].x + ShearMatrix[3 * N + i].x ) );
    }

}



// compute pseudo-pressure and subtract grad p_tilde from dfields
__global__ void GradPseudoPressure(const scalar_type *d_all_kvec, cufftDoubleComplex *d_all_dfields, size_t N){
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    cufftDoubleComplex divDeltaField;
    scalar_type ik2 = 1.0;
    // int KX = 0; int KY = 1; int KZ = 2;
    // int VX = 0; int VY = 1; int VZ = 2;
    if (i < N) {
        // real part
        divDeltaField.x = - d_all_kvec[KX * N + i] * d_all_dfields[VX * N + i].y - d_all_kvec[KY * N + i] * d_all_dfields[VY * N + i].y - d_all_kvec[KZ * N + i] * d_all_dfields[VZ * N + i].y ;
        // complex part
        divDeltaField.y =   d_all_kvec[KX * N + i] * d_all_dfields[VX * N + i].x + d_all_kvec[KY * N + i] * d_all_dfields[VY * N + i].x + d_all_kvec[KZ * N + i] * d_all_dfields[VZ * N + i].x ;

        // compute 1/k2
        if (i > 0){
            ik2 = 1.0 / (d_all_kvec[KX * N + i] * d_all_kvec[KX * N + i] + d_all_kvec[KY * N + i] * d_all_kvec[KY * N + i] + d_all_kvec[KZ * N + i] * d_all_kvec[KZ * N + i]);
        }

        // add -grad p
        // vx component
        d_all_dfields[VX * N + i].x += - d_all_kvec[KX * N + i] * ik2 * divDeltaField.y;
        d_all_dfields[VX * N + i].y +=   d_all_kvec[KX * N + i] * ik2 * divDeltaField.x;

        // vy component
        d_all_dfields[VY * N + i].x += - d_all_kvec[KY * N + i] * ik2 * divDeltaField.y;
        d_all_dfields[VY * N + i].y +=   d_all_kvec[KY * N + i] * ik2 * divDeltaField.x;

        // vz component
        d_all_dfields[VZ * N + i].x += - d_all_kvec[KZ * N + i] * ik2 * divDeltaField.y;
        d_all_dfields[VZ * N + i].y +=   d_all_kvec[KZ * N + i] * ik2 * divDeltaField.x;

    }

}

#endif



#ifdef MHD

// compute the elements of the traceless symmetric matrix in the MHD case T_ij = u_i u_j  - delta_ij Tr (u_i u_j) / 3 - B_i B_j + delta_ij Tr (B_i B_j) / 3. It has only 5 independent components T_xx, T_xy, T_xz, Tyy, T_yz. (T_zz = - T_xx - T_yy)
// the results are saved in the first 5 temp_arrays (after those reserved for the fields, the memory block points already at the right location)
__global__ void TracelessShearMatrixMHD( const scalar_type *d_all_fields, scalar_type *d_all_tmparray, size_t N) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // int VX = 0; int VY = 1; int VZ = 2;
    if (i < N) {

        // 0: T_xx = u_x^2 - u^2/3 - B_x^2 + B^2/3 = (2./3.) ( u_x^2 - B_x^2) - (1./3.) ( u_y^2 - B_y^2 + u_z^2 - B_z^2 )
        d_all_tmparray[         i] = ( 2.0 * d_all_fields[ VX * N + i] * d_all_fields[ VX * N + i] - d_all_fields[ VY            * N + i] * d_all_fields[ VY * N + i] - d_all_fields[ VZ * N + i] * d_all_fields[ VZ * N + i]
        - 2.0 * d_all_fields[ BX * N + i] * d_all_fields[ BX * N + i] + d_all_fields[ BY * N + i] * d_all_fields[ BY * N + i] + d_all_fields[ BZ * N + i] * d_all_fields[ BZ * N + i] ) / 3.0;
        // 1: T_xy = u_x u_y - B_x B_y
        d_all_tmparray[N      + i] = d_all_fields[ VX * N + i] * d_all_fields[ VY * N + i] - d_all_fields[ BX * N + i] * d_all_fields[ BY * N + i] ;
        // 2: T_xz = u_x u_z - B_x B_z
        d_all_tmparray[ 2 * N + i] = d_all_fields[ VX * N + i] * d_all_fields[ VZ * N + i] - d_all_fields[ BX * N + i] * d_all_fields[ BZ * N + i] ;
        // 3: T_yy = u_y^2 - u^2/3 - B_y^2 + B^2/3 = - 1./3 ( u_x^2 - B_x^3 ) + 2./3. ( u_y^2 - B_y^2 ) - 1./3. ( u_z^2 - B_z^2)
        d_all_tmparray[ 3 * N + i] = ( - d_all_fields[ VX * N + i] * d_all_fields[ VX * N + i] + 2.0 * d_all_fields[ VY * N + i] * d_all_fields[ VY * N + i] - d_all_fields[ VZ * N + i] * d_all_fields[ VZ * N + i]
        + d_all_fields[ BX * N + i] * d_all_fields[ BX * N + i] - 2.0 * d_all_fields[ BY * N + i] * d_all_fields[ BY * N + i] + d_all_fields[ BZ * N + i] * d_all_fields[ BZ * N + i] ) / 3.0;
        // 4: T_yz = u_y u_z - B_y B_z
        d_all_tmparray[ 4 * N + i] = d_all_fields[ VY * N + i] * d_all_fields[ VZ * N + i] - d_all_fields[ BY * N + i] * d_all_fields[ BZ * N + i];
    }
}


// compute emf = u x B:
// emf_x = u_y B_z - u_z B_y , emf_y = u_z B_x - u_x B_z , emf_z = u_x B_y - u_y B_x
// the results are saved in the first 3 temp_arrays (after those reserved for the fields, the memory block points already at the right location) as [emf_x, emf_y, emf_z] (they are the x,y,z components of the emf)
__global__ void MagneticEmf( const scalar_type *d_all_fields, scalar_type *d_all_tmparray, size_t N) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // int VX = 0; int VY = 1; int VZ = 2, int TH = 3;
    if (i < N) {

        // 2: emf_z = u_x B_y - u_y B_x
        d_all_tmparray[ 2 * N + i] = - d_all_fields[ BX * N + i] * d_all_fields[ VY * N + i] + d_all_fields[ BY * N + i] * d_all_fields[ VX * N + i] ;
        // 1: emf_y = u_z B_x - u_x B_z
        d_all_tmparray[     N + i] =   d_all_fields[ BX * N + i] * d_all_fields[ VZ * N + i] - d_all_fields[ BZ * N + i] * d_all_fields[ VX * N + i] ;
        // 0: emf_x = u_y B_z - u_z B_y
        d_all_tmparray[         i] = - d_all_fields[ BY * N + i] * d_all_fields[ VZ * N + i] + d_all_fields[ BZ * N + i] * d_all_fields[ VY * N + i] ;
    }
}


// compute derivatives of emf and assign to magnetic fields
// curl of emf is in the first 3 temp_arrays (after those reserved for the fields, the memory block points already at the right location)
__global__ void MagneticShear(const scalar_type *d_all_kvec, const cufftDoubleComplex *MagEmf, cufftDoubleComplex *d_all_dfields, const scalar_type *d_mask, size_t N){
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // int KX = 0; int KY = 1; int KZ = 2;
    // int VX = 0; int VY = 1; int VZ = 2;
    if (i < N) {
        // delta B_x =  ( I k_y emf_z - I k_z emf_y )
        // operations divided in real and complex part
        d_all_dfields[BX * N + i].x =  - d_mask[i] * ( d_all_kvec[KY * N + i] * MagEmf[2 * N + i].y - d_all_kvec[KZ * N + i] * MagEmf[    N + i].y );
        d_all_dfields[BX * N + i].y =    d_mask[i] * ( d_all_kvec[KY * N + i] * MagEmf[2 * N + i].x - d_all_kvec[KZ * N + i] * MagEmf[    N + i].x );

        // delta B_y =  ( I k_z emf_x - I k_x emf_z )
        // operations divided in real and complex part
        d_all_dfields[BY * N + i].x =  - d_mask[i] * ( d_all_kvec[KZ * N + i] * MagEmf[        i].y - d_all_kvec[KX * N + i] * MagEmf[2 * N + i].y );
        d_all_dfields[BY * N + i].y =    d_mask[i] * ( d_all_kvec[KZ * N + i] * MagEmf[        i].x - d_all_kvec[KX * N + i] * MagEmf[2 * N + i].x );

        // delta B_z =  ( I k_x emf_y - I k_y emf_x )
        // operations divided in real and complex part
        d_all_dfields[BZ * N + i].x =  - d_mask[i] * ( d_all_kvec[KX * N + i] * MagEmf[1 * N + i].y - d_all_kvec[KY * N + i] * MagEmf[        i].y );
        d_all_dfields[BZ * N + i].y =    d_mask[i] * ( d_all_kvec[KX * N + i] * MagEmf[1 * N + i].x - d_all_kvec[KY * N + i] * MagEmf[        i].x  );
    }

}


#endif



#ifdef BOUSSINESQ

// compute the elements of the energy flux vector matrix E_i = u_i x theta. It has 3 independent components E_x, E_y, E_z
// the results are saved in the first 3 temp_arrays (after those reserved for the fields, the memory block points already at the right location)
__global__ void EnergyFluxVector( const scalar_type *d_all_fields, scalar_type *d_all_tmparray, size_t N) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // int VX = 0; int VY = 1; int VZ = 2, int TH = 3;
    if (i < N) {

        // 0: E_x = u_x  theta
        d_all_tmparray[         i] = d_all_fields[ VX * N + i] * d_all_fields[ TH * N + i] ;
        // 1: E_y = u_y  theta
        d_all_tmparray[     N + i] = d_all_fields[ VY * N + i] * d_all_fields[ TH * N + i] ;
        // 2: E_z = u_z  theta
        d_all_tmparray[ 2 * N + i] = d_all_fields[ VZ * N + i] * d_all_fields[ TH * N + i] ;
    }
}


// compute derivative of energy flux vector and assign u nabla theta to the dfield for theta
__global__ void NonLinBoussinesqAdv(const scalar_type *d_all_kvec, const cufftDoubleComplex *EnergyFlux, cufftDoubleComplex *d_all_dfields, const scalar_type *d_mask, size_t N){
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // int KX = 0; int KY = 1; int KZ = 2;
    // int VX = 0; int VY = 1; int VZ = 2;
    if (i < N) {
        // delta theta = - ( I k_x E_x + I k_y E_y + I k_z E_z)
        // operations divided in real and complex part
        d_all_dfields[TH * N + i].x =   d_mask[i] * (  d_all_kvec[KX * N + i] * EnergyFlux[i].y + d_all_kvec[KY * N + i] * EnergyFlux[N + i].y + d_all_kvec[KZ * N + i] * EnergyFlux[2 * N + i].y );
        d_all_dfields[TH * N + i].y =   d_mask[i] * (- d_all_kvec[KX * N + i] * EnergyFlux[i].x - d_all_kvec[KY * N + i] * EnergyFlux[N + i].x - d_all_kvec[KZ * N + i] * EnergyFlux[2 * N + i].x );

    }

}

// add - th e_strat to velocity component in the strat direction
// add N2 u_strat to temperature equation
// this is for normalization where theta is in units of g [L/T^2]
// other normalizations possible
__global__ void BoussinesqStrat( const cufftDoubleComplex *d_all_fields, cufftDoubleComplex *d_all_dfields, double BV_freq2, size_t N, int strat_dir){
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N) {
        // add - th e_strat to velocity component in the strat direction
        // strat_dir can be 0 (e_x), 1 (e_y), 2 (e_z) and is defined in common.hpp
        // operations divided in real and complex part
        d_all_dfields[strat_dir * N + i].x +=   - d_all_fields[TH * N + i].x ;
        d_all_dfields[strat_dir * N + i].y +=   - d_all_fields[TH * N + i].y ;

        // add N2 u_strat to temperature equation
        // BV_freq2 is the squared BV frequency (can be negative)
        // operations divided in real and complex part
        d_all_dfields[TH * N + i].x +=   BV_freq2 * d_all_fields[strat_dir * N + i].x ;
        d_all_dfields[TH * N + i].y +=   BV_freq2 * d_all_fields[strat_dir * N + i].y ;
    }
}

#endif
