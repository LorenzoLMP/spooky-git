#include <cuda_runtime.h>
#include <cufftXt.h>
#include "cufft_utils.h"
// #include "cufft_routines.hpp"
#include "common.hpp"

/*
 * In the following kernels it is always
 * assumed that:
 *
 * vars.KX = 0
 * vars.KY = 1
 * vars.KZ = 2
 *
 * Therefore:
 *
 * kvec[vars.KX * N + i] = i-th element of KX = kvec[0 * N + i]
 * kvec[vars.KY * N + i] = i-th element of KY = kvec[1 * N + i]
 * ...
*/

/*
 * In the following kernels it is
 * assumed that:
 *
 * vars.VX = 0
 * vars.VY = 1
 * vars.VZ = 2
 *
 * Therefore:
 *
 * VelField[vars.VX * N + i] = i-th element of VX = VelField[0 * N + i]
 * VelField[vars.VY * N + i] = i-th element of VY = VelField[1 * N + i]
 * ...
*/

// compute the elements of the traceless symmetric matrix S_ij = u_i u_j - delta_ij Tr (u_i u_j) / 3. It has only 5 independent components S_xx, S_xy, S_xz, Byy, S_yz. (S_zz = - S_xx - S_yy)
// the results are saved in 5 temp arrays (the memory block ShearMatrix points already at the right location)
__global__ void TracelessShearMatrix( const scalar_type *VelField, scalar_type *ShearMatrix, size_t N) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {

        // 0: S_xx = u_x^2 - u^2/3
        ShearMatrix[ 0 * N + i] = ( 2.0 * VelField[ 0 * N + i] * VelField[ 0 * N + i] - VelField[ 1 * N + i] * VelField[ 1 * N + i] - VelField[ 2 * N + i] * VelField[ 2 * N + i] ) / 3.0;
        // 1: S_xy = u_x u_y
        ShearMatrix[ 1 * N + i] = VelField[ 0 * N + i] * VelField[ 1 * N + i] ;
        // 2: S_xz = u_x u_z
        ShearMatrix[ 2 * N + i] = VelField[ 0 * N + i] * VelField[ 2 * N + i] ;
        // 3: S_yy = u_y^2 - u^2/3
        ShearMatrix[ 3 * N + i] = ( - VelField[ 0 * N + i] * VelField[ 0 * N + i] + 2.0 * VelField[ 1 * N + i] * VelField[ 1 * N + i] - VelField[ 2 * N + i] * VelField[ 2 * N + i] ) / 3.0;
        // 4: S_yz = u_y u_z
        ShearMatrix[ 4 * N + i] = VelField[ 1 * N + i] * VelField[ 2 * N + i] ;
    }
}



// compute derivative of traceless shear matrix and assign to dfields
__global__ void NonLinHydroAdv(const scalar_type *kvec, const data_type *ShearMatrix, data_type *dVelField, const scalar_type *mask, size_t N){
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // this is the imaginary unit
    data_type imI = data_type(0.0,1.0);
    if (i < N) {
        // delta u_x = - ( I k_x S_xx + I k_y S_xy + I k_z S_xz)
        dVelField[0 * N + i] = - imI * mask[i] * (  kvec[0 * N + i] * ShearMatrix[i] + kvec[1 * N + i] * ShearMatrix[N + i] + kvec[2 * N + i] * ShearMatrix[2 * N + i] );

        // delta u_y = - ( I k_x S_yx + I k_y S_yy + I k_z S_yz)
        dVelField[1 * N + i] = - imI * mask[i] * (  kvec[0 * N + i] * ShearMatrix[N + i] + kvec[1 * N + i] * ShearMatrix[3 * N + i] + kvec[2 * N + i] * ShearMatrix[4 * N + i] );

        // delta u_z = - ( I k_x S_zx + I k_y S_zy + I k_z S_zz)
        // note that S_zz = - S_xx - S_yy
        dVelField[2 * N + i] =  - imI * mask[i] * (  kvec[0 * N + i] * ShearMatrix[2 * N + i] + kvec[1 * N + i] * ShearMatrix[4 * N + i] - kvec[2 * N + i] * ( ShearMatrix[i] + ShearMatrix[3 * N + i] ) );

    }
}



// compute pseudo-pressure and subtract grad p_tilde from dfields
__global__ void GradPseudoPressure(const scalar_type *kvec, data_type *dVelField, size_t N){
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    data_type divDeltaField;
    scalar_type ik2 = 1.0;
    // this is the imaginary unit
    data_type imI = data_type(0.0,1.0);

    if (i < N) {
        divDeltaField = imI * ( kvec[0 * N + i] * dVelField[0 * N + i] + kvec[1 * N + i] * dVelField[1 * N + i] + kvec[2 * N + i] * dVelField[2 * N + i] ) ;

        // compute 1/k2
        if (i > 0){
            ik2 = 1.0 / (kvec[0 * N + i] * kvec[0 * N + i] + kvec[1 * N + i] * kvec[1 * N + i] + kvec[2 * N + i] * kvec[2 * N + i]);
        }

        // add -grad p
        // vx component
        dVelField[0 * N + i] += imI * kvec[0 * N + i] * ik2 * divDeltaField;

        // vy component
        dVelField[1 * N + i] += imI * kvec[1 * N + i] * ik2 * divDeltaField;

        // vz component
        dVelField[2 * N + i] += imI * kvec[2 * N + i] * ik2 * divDeltaField;

    }

}


// compute pseudo-pressure and subtract grad p_tilde from dfields
// with background shearing ON

// need to finish this

__global__ void GradPseudoPressureShearing(const scalar_type *kvec, data_type *dVelField, size_t N){
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    data_type divDeltaField;
    scalar_type ik2 = 1.0;
    // this is the imaginary unit
    data_type imI = data_type(0.0,1.0);

    if (i < N) {
        divDeltaField = imI * ( kvec[0 * N + i] * dVelField[0 * N + i] + kvec[1 * N + i] * dVelField[1 * N + i] + kvec[2 * N + i] * dVelField[2 * N + i] ) ;

        // compute 1/k2
        if (i > 0){
            ik2 = 1.0 / (kvec[0 * N + i] * kvec[0 * N + i] + kvec[1 * N + i] * kvec[1 * N + i] + kvec[2 * N + i] * kvec[2 * N + i]);
        }

        // add -grad p
        // vx component
        dVelField[0 * N + i] += imI * kvec[0 * N + i] * ik2 * divDeltaField;

        // vy component
        dVelField[1 * N + i] += imI * kvec[1 * N + i] * ik2 * divDeltaField;

        // vz component
        dVelField[2 * N + i] += imI * kvec[2 * N + i] * ik2 * divDeltaField;

    }

}

/*
 * In the following kernels it is
 * assumed that:
 *
 * vars.BX = 0
 * vars.BY = 1
 * vars.BZ = 2
 *
 * Therefore:
 *
 * MagField[vars.BX * N + i] = i-th element of BX = MagField[0 * N + i]
 *
 * ...
*/

// compute the elements of the traceless symmetric matrix in the MHD case T_ij = u_i u_j  - delta_ij Tr (u_i u_j) / 3 - B_i B_j + delta_ij Tr (B_i B_j) / 3. It has only 5 independent components T_xx, T_xy, T_xz, Tyy, T_yz. (T_zz = - T_xx - T_yy)
// the results are saved in 5 temp arrays (the memory block TShearMatrix points already at the right location)
__global__ void TracelessShearMatrixMHD( const scalar_type *VelField, const scalar_type *MagField, scalar_type *TShearMatrix, size_t N) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {

        // 0: T_xx = u_x^2 - u^2/3 - B_x^2 + B^2/3 = (2./3.) ( u_x^2 - B_x^2) - (1./3.) ( u_y^2 - B_y^2 + u_z^2 - B_z^2 )
        TShearMatrix[         i] = ( 2.0 * VelField[ 0 * N + i] * VelField[ 0 * N + i] - VelField[ 1            * N + i] * VelField[ 1 * N + i] - VelField[ 2 * N + i] * VelField[ 2 * N + i]
        - 2.0 * MagField[ 0 * N + i] * MagField[ 0 * N + i] + MagField[ 1 * N + i] * MagField[ 1 * N + i] + MagField[ 2 * N + i] * MagField[ 2 * N + i] ) / 3.0;
        // 1: T_xy = u_x u_y - B_x B_y
        TShearMatrix[N      + i] = VelField[ 0 * N + i] * VelField[ 1 * N + i] - MagField[ 0 * N + i] * MagField[ 1 * N + i] ;
        // 2: T_xz = u_x u_z - B_x B_z
        TShearMatrix[ 2 * N + i] = VelField[ 0 * N + i] * VelField[ 2 * N + i] - MagField[ 0 * N + i] * MagField[ 2 * N + i] ;
        // 3: T_yy = u_y^2 - u^2/3 - B_y^2 + B^2/3 = - 1./3 ( u_x^2 - B_x^3 ) + 2./3. ( u_y^2 - B_y^2 ) - 1./3. ( u_z^2 - B_z^2)
        TShearMatrix[ 3 * N + i] = ( - VelField[ 0 * N + i] * VelField[ 0 * N + i] + 2.0 * VelField[ 1 * N + i] * VelField[ 1 * N + i] - VelField[ 2 * N + i] * VelField[ 2 * N + i]
        + MagField[ 0 * N + i] * MagField[ 0 * N + i] - 2.0 * MagField[ 1 * N + i] * MagField[ 1 * N + i] + MagField[ 2 * N + i] * MagField[ 2 * N + i] ) / 3.0;
        // 4: T_yz = u_y u_z - B_y B_z
        TShearMatrix[ 4 * N + i] = VelField[ 1 * N + i] * VelField[ 2 * N + i] - MagField[ 1 * N + i] * MagField[ 2 * N + i];
    }
}


// compute emf = u x B:
// emf_x = u_y B_z - u_z B_y , emf_y = u_z B_x - u_x B_z , emf_z = u_x B_y - u_y B_x
// the results are saved in the first 3 temp_arrays (after those reserved for the fields, the memory block points already at the right location) as [emf_x, emf_y, emf_z] (they are the x,y,z components of the emf)
__global__ void MagneticEmf( const scalar_type *VelField, const scalar_type *MagField, scalar_type *Emf, size_t N) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {

        // 0: emf_x = u_y B_z - u_z B_y
        Emf[         i] = - MagField[ 1 * N + i] * VelField[ 2 * N + i] + MagField[ 2 * N + i] * VelField[ 1 * N + i] ;
        // 1: emf_y = u_z B_x - u_x B_z
        Emf[     N + i] =   MagField[ 0 * N + i] * VelField[ 2 * N + i] - MagField[ 2 * N + i] * VelField[ 0 * N + i] ;
        // 2: emf_z = u_x B_y - u_y B_x
        Emf[ 2 * N + i] = - MagField[ 0 * N + i] * VelField[ 1 * N + i] + MagField[ 1 * N + i] * VelField[ 0 * N + i] ;
    }
}


// compute derivatives of emf and assign to magnetic fields
// curl of emf is in the first 3 temp_arrays (after those reserved for the fields, the memory block points already at the right location)
__global__ void MagneticShear(const scalar_type *kvec, const data_type *MagEmf, data_type *dMagField, const scalar_type *mask, size_t N){
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // this is the imaginary unit
    data_type imI = data_type(0.0,1.0);
    if (i < N) {
        // delta B_x =  ( I k_y emf_z - I k_z emf_y )
        // operations divided in real and complex part
        dMagField[0 * N + i] =  imI * mask[i] * ( kvec[1 * N + i] * MagEmf[2 * N + i] - kvec[2 * N + i] * MagEmf[    N + i] );

        // delta B_y =  ( I k_z emf_x - I k_x emf_z )
        // operations divided in real and complex part
        dMagField[1 * N + i] =  imI * mask[i] * ( kvec[2 * N + i] * MagEmf[        i] - kvec[0 * N + i] * MagEmf[2 * N + i] );

        // delta B_z =  ( I k_x emf_y - I k_y emf_x )
        // operations divided in real and complex part
        dMagField[2 * N + i] =  imI * mask[i] * ( kvec[0 * N + i] * MagEmf[1 * N + i] - kvec[1 * N + i] * MagEmf[        i] );
    }

}


// compute the elements of the energy flux vector matrix E_i = u_i x theta. It has 3 independent components E_x, E_y, E_z
// the results are saved in the first 3 temp_arrays (after those reserved for the fields, the memory block points already at the right location)
__global__ void EnergyFluxVector( const scalar_type *VelField, const scalar_type *Theta, scalar_type *EnFlux, size_t N) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {

        // 0: E_x = u_x  theta
        EnFlux[         i] = VelField[ 0 * N + i] * Theta[i] ;
        // 1: E_y = u_y  theta
        EnFlux[     N + i] = VelField[ 1 * N + i] * Theta[i] ;
        // 2: E_z = u_z  theta
        EnFlux[ 2 * N + i] = VelField[ 2 * N + i] * Theta[i] ;
    }
}


// compute derivative of energy flux vector and assign u nabla theta to the dfield for theta
__global__ void NonLinBoussinesqAdv(const scalar_type *kvec, const data_type *EnergyFlux, data_type *dTheta, const scalar_type *mask, size_t N){
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // this is the imaginary unit
    data_type imI = data_type(0.0,1.0);
    if (i < N) {
        // delta theta = - ( I k_x E_x + I k_y E_y + I k_z E_z)
        dTheta[i] = - imI * mask[i] * (  kvec[0 * N + i] * EnergyFlux[i] + kvec[1 * N + i] * EnergyFlux[N + i] + kvec[2 * N + i] * EnergyFlux[2 * N + i]);


    }

}

// add - th e_strat to velocity component in the strat direction
// add N2 u_strat to temperature equation
// this is for normalization where theta is in units of g [L/T^2]
// other normalizations possible
__global__ void BoussinesqStrat( const data_type *VelField, const data_type *Theta, data_type *dVelField, data_type *dTheta, double BV_freq2, size_t N, int strat_dir){
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // this is the imaginary unit
    data_type imI = data_type(0.0,1.0);

    if (i < N) {
        // add - th e_strat to velocity component in the strat direction
        // strat_dir can be 0 (e_x), 1 (e_y), 2 (e_z) and is defined in common.hpp
        dVelField[strat_dir * N + i] +=   - Theta[i] ;

        // add N2 u_strat to temperature equation
        // BV_freq2 is the squared BV frequency (can be negative)
        dTheta[i] +=   BV_freq2 * VelField[strat_dir * N + i] ;

    }
}


// add du_y += param_ptr->shear * u_x
// and dB_y -= param_ptr->shear * B_x (if MHD)
// note the sign difference when the kernel is called in shear_rotation.cu

__global__ void ShearingFlow( const data_type *complex_Vecx, data_type *complex_dVecy,  double shear, size_t N){
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // this is the imaginary unit
    // data_type imI = data_type(0.0,1.0);

    if (i < N) {
        complex_dVecy[i] += shear * complex_Vecx[i] ;
    }
}

// add du_x += 2.0 * param_ptr->omega * u_y
// add du_y -= 2.0 * param_ptr->omega * u_x
__global__ void CoriolisForce( const data_type *complex_Vecx, const data_type *complex_Vecy, data_type *complex_dVecx, data_type *complex_dVecy, double omega, size_t N) {

    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // this is the imaginary unit
    // data_type imI = data_type(0.0,1.0);

    if (i < N) {
        complex_dVecx[i] += 2.0 * omega * complex_Vecy[i] ;
        complex_dVecy[i] -= 2.0 * omega * complex_Vecx[i] ;
    }

}


// compute the scalar B \cdot grad theta (in real space), and assign it to a new scratch array
// B points at the memory location of the 3D vector, GradTheta points at the memory location of the 3D grad theta, Z is the scratch array
__global__ void ComputeBGradTheta( const scalar_type *B, const scalar_type *GradTheta, scalar_type *Z, size_t N) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N) {

        // Z = Bx gradx theta + By grady theta + Bz gradz theta
        Z[i] = B[ 0 * N + i ] * GradTheta[ 0 * N + i ]  +  B[ 1 * N + i ] * GradTheta[ 1 * N + i ] + B[ 2 * N + i ] * GradTheta[ 2 * N + i ] ;
    }
}

// compute the vector (b b \cdot grad theta + b b_z) (in real space), and assign it to a new scratch 3D array
// B points at the memory location of the 3D vector, BGradTheta points at the memory location of the scalar field, Z is the 3D scratch array
__global__ void ComputeAnisotropicHeatFlux( const scalar_type *B, const scalar_type *BGradTheta, scalar_type *Z, scalar_type OmegaT2, scalar_type chi, size_t N, int strat_dir) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    scalar_type B2 = 0.0;
    if (i < N) {
        B2 = B[ 0 * N + i ] * B[ 0 * N + i ] + B[ 1 * N + i ] * B[ 1 * N + i ] + B[ 2 * N + i ] * B[ 2 * N + i ] ;
        // Z_x = B_x BGradTheta / B^2
        Z[ 0 * N + i ] = chi * B[ 0 * N + i ] * ( BGradTheta[ i ] + OmegaT2 * B[ strat_dir * N + i ] ) / B2 ;
        // Z_y = B_y BGradTheta / B^2
        Z[ 1 * N + i ] = chi * B[ 1 * N + i ] * ( BGradTheta[ i ] + OmegaT2 * B[ strat_dir * N + i ] ) / B2 ;
        // Z_z = B_z BGradTheta / B^2
        Z[ 2 * N + i ] = chi * B[ 2 * N + i ] * ( BGradTheta[ i ] + OmegaT2 * B[ strat_dir * N + i ] ) / B2 ;
    }
}

