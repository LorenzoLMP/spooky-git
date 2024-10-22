#include "define_types.hpp"
#include "physics.hpp"
#include "fields.hpp"
#include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "parameters.hpp"

void Physics::Boussinesq(Fields &fields, Parameters &param) {

    int blocksPerGrid;

#ifdef BOUSSINESQ
    // first compute energy flux vector [ u_x theta, u_y theta, u_z theta]
    // we can re-utilize tmparrays and store result in tmparray_r[num_fields] - tmparray_r[num_fields + 3]
    blocksPerGrid = ( 2 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    EnergyFluxVector<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)fields.d_all_tmparray, (scalar_type *)fields.d_all_tmparray + 2 * ntotal_complex * fields.num_fields,  2 * ntotal_complex);


    // take fourier transforms of the 3 energy flux vector components
    for (int n = fields.num_fields ; n < fields.num_fields + 3; n++) {
        r2c_fft(fields.d_tmparray_r[n], fields.d_tmparray[n], supervisor);
    }


    // compute derivative of energy flux vector and assign u nabla theta to the dfield for theta
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    NonLinBoussinesqAdv<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)fields.wavevector.d_all_kvec, (data_type *)fields.d_all_tmparray + ntotal_complex * fields.num_fields, (data_type *) fields.d_all_dfields, (scalar_type *)fields.wavevector.d_mask, ntotal_complex);



#ifdef STRATIFICATION
    // add - th e_strat to velocity component in the strat direction
    // add N2 u_strat to temperature equation
    // this is for normalization where theta is in units of g [L/T^2]
    // other normalizations possible
    EntropyStratification(fields, param);
#endif

/*
 *
 * Here we do the diffusion terms
 *
 */

#ifndef ANISOTROPIC_DIFFUSION
    //  for explicit treatment of energy diffusion term
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    nablaOpScalar<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *) fields.wavevector.d_all_kvec, (data_type *) fields.d_farray[TH], (data_type *) fields.d_dfarray[TH], param.nu_th, (size_t) ntotal_complex, ADD);
#else
#ifdef MHD
    // AnisotropicConduction(fields, param);
    AnisotropicConduction(fields, param, (data_type *) fields.d_farray[TH], (data_type *) fields.d_dfarray[TH]);

    /*
    // assign Bx, By, Bz to first 3 scratch arrays
    blocksPerGrid = ( 3 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    ComplexVecAssign<<<blocksPerGrid, threadsPerBlock>>>((cufftDoubleComplex *)d_all_fields + ntotal_complex * BX, (cufftDoubleComplex *)d_all_tmparray, 3 * ntotal_complex);
    // compute gradient of theta and assign it to next 3 scratch arrays
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    Gradient<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)wavevector.d_all_kvec, (data_type *) d_farray[TH], (data_type *)d_all_tmparray + 3 * ntotal_complex, ntotal_complex);
    // compute complex to real iFFTs
    for (int n = 0; n < 6; n++){
        c2r_fft(d_tmparray[n], d_tmparray_r[n]);
    }
    // compute the scalar B grad theta (real space) and assign it to 7th scratch array
    blocksPerGrid = ( 2 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    ComputeBGradTheta<<<blocksPerGrid, threadsPerBlock>>>( (scalar_type *) d_tmparray_r[0], (scalar_type *) d_tmparray_r[3], (scalar_type *) d_tmparray_r[6], 2 * ntotal_complex);
    // compute the anisotropic heat flux and put it in the 3-4-5 tmp arrays
    ComputeAnisotropicHeatFlux<<<blocksPerGrid, threadsPerBlock>>>( (scalar_type *) d_tmparray_r[0], (scalar_type *) d_tmparray_r[6], (scalar_type *) d_tmparray_r[3], param.OmegaT2, (1./param.reynolds_ani), 2 * ntotal_complex, STRAT_DIR);
    // take fourier transforms of the heat flux
    for (int n = 3 ; n < 6; n++) {
        r2c_fft(d_tmparray_r[n], d_tmparray[n]);
    }
    // take divergence of heat flux
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    DivergenceMask<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)wavevector.d_all_kvec, (data_type *) d_tmparray[3], (data_type *) d_all_dfields + TH * ntotal_complex, (scalar_type *)wavevector.d_mask, ntotal_complex, ADD);
    */
#endif   // MHD
#endif   // ANISOTROPIC_DIFFUSION
#endif // Boussinesq


}
