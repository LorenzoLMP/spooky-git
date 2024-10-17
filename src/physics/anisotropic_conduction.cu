#include "define_types.hpp"
#include "physics.hpp"
#include "fields.hpp"
#include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "parameters.hpp"

void Physics::AnisotropicConduction(Fields &fields, Parameters &param) {

    int blocksPerGrid;

#ifdef BOUSSINESQ
#ifdef MHD
#ifdef ANISOTROPIC_DIFFUSION

    // not needed anymore!
    // assign Bx, By, Bz to first 3 scratch arrays
    // blocksPerGrid = ( 3 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    // ComplexVecAssign<<<blocksPerGrid, threadsPerBlock>>>((cufftDoubleComplex *)d_all_fields + ntotal_complex * BX, (cufftDoubleComplex *)d_all_tmparray, 3 * ntotal_complex);


    // Bx, By, Bz real fields are already in the 4-5-6 tmp arrays
    // compute gradient of theta and assign it to next 3 scratch arrays [num_fields -- num_fields + 3] (the first num_fields arrays are reserved for the real-valued fields)
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    Gradient<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)fields.wavevector.d_all_kvec, (data_type *) fields.d_farray[TH], (data_type *)fields.d_all_tmparray + fields.num_fields * ntotal_complex, ntotal_complex);
    // compute complex to real iFFTs
    for (int n = fields.num_fields; n < fields.num_fields + 3; n++){
        c2r_fft(fields.d_tmparray[n], fields.d_tmparray_r[n], supervisor);
    }
    // compute the scalar B grad theta (real space) and assign it to 7th scratch array
    blocksPerGrid = ( 2 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    ComputeBGradTheta<<<blocksPerGrid, threadsPerBlock>>>( (scalar_type *) fields.d_tmparray_r[BX], (scalar_type *) fields.d_all_tmparray + 2 * ntotal_complex * fields.num_fields, (scalar_type *) fields.d_tmparray_r[fields.num_fields + 3], 2 * ntotal_complex);
    // compute the anisotropic heat flux and put it in the 3-4-5 tmp arrays
    ComputeAnisotropicHeatFlux<<<blocksPerGrid, threadsPerBlock>>>( (scalar_type *) fields.d_tmparray_r[BX], (scalar_type *) fields.d_tmparray_r[fields.num_fields + 3], (scalar_type *) fields.d_tmparray_r[fields.num_fields], param.OmegaT2, (1./param.reynolds_ani), 2 * ntotal_complex, STRAT_DIR);
    // take fourier transforms of the heat flux
    for (int n = fields.num_fields ; n < fields.num_fields + 3; n++) {
        r2c_fft(fields.d_tmparray_r[n], fields.d_tmparray[n], supervisor);
    }
    // take divergence of heat flux
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    DivergenceMask<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)fields.wavevector.d_all_kvec, (data_type *) fields.d_tmparray[fields.num_fields], (data_type *) fields.d_all_dfields + TH * ntotal_complex, (scalar_type *)fields.wavevector.d_mask, ntotal_complex, ADD);

#endif // ANISOTROPIC_DIFFUSION
#endif // MHD
#endif // BOUSSINESQ

}
