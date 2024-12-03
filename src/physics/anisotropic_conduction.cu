#include "define_types.hpp"
#include "physics.hpp"
#include "fields.hpp"
#include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "parameters.hpp"
#include "supervisor.hpp"

void Physics::AnisotropicConduction(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dTemp) {

    int blocksPerGrid;

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;
    scalar_type* mask = fields_ptr->wavevector.d_mask;

#ifdef BOUSSINESQ
#ifdef MHD
#ifdef ANISOTROPIC_DIFFUSION

    // compute gradient of theta and assign it to the first 3 scratch arrays [0, 1, 2]
    data_type* grad_theta = fields_ptr->d_all_tmparray;

    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    Gradient<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_Fields + ntotal_complex * TH, grad_theta, ntotal_complex);
    // compute complex to real iFFTs
    for (int n = 0; n < 3; n++){
        c2r_fft(grad_theta + n*ntotal_complex, ((scalar_type *) grad_theta) + 2*n*ntotal_complex , supervisor_ptr);
    }


    // compute the scalar B grad theta (real space) and assign it to [3] scratch array
    // Bx, By, Bz real fields are already in the 4-5-6 real_Buffer arrays
    // NOTE: for supertimestepping strictly speaking this is not 100% self-consistent because the real magnetic field in d_tmparray_r refer to those prior to the supertimestepping, so in theory before proceeding they would  need to be computed again from fields_in. However, with an isotropic magnetic resistivity, the value of b_x, b_y, b_z do not change during supertimestepping (even though B_x, B_y and B_z do). So here we are good, because \bm b is all that matters for the anisotropic conduction.

    scalar_type* mag_vec = real_Buffer + 2 * ntotal_complex * BX;
    // scalar_type* mag_vec = fields_ptr->d_tmparray_r[BX];

    scalar_type* Bgrad_theta = fields_ptr->d_tmparray_r[3];

    blocksPerGrid = ( 2 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    ComputeBGradTheta<<<blocksPerGrid, threadsPerBlock>>>(mag_vec, (scalar_type *) grad_theta, Bgrad_theta, 2 * ntotal_complex);

    // compute the anisotropic heat flux and put it in the [0, 1, 2] tmp array
    scalar_type* heat_flux = fields_ptr->d_tmparray_r[0];

    ComputeAnisotropicHeatFlux<<<blocksPerGrid, threadsPerBlock>>>( mag_vec, Bgrad_theta, heat_flux, param_ptr->OmegaT2, (1./param_ptr->reynolds_ani), 2 * ntotal_complex, STRAT_DIR);

    // take fourier transforms of the heat flux
    for (int n = 0 ; n < 3; n++) {
        r2c_fft(heat_flux + 2*n*ntotal_complex, ((data_type*) heat_flux) + n*ntotal_complex, supervisor_ptr);
    }

    // take divergence of heat flux and add to dtemp
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    DivergenceMask<<<blocksPerGrid, threadsPerBlock>>>(kvec, (data_type*) heat_flux, complex_dTemp, mask, ntotal_complex, ADD);

#endif // ANISOTROPIC_DIFFUSION
#endif // MHD
#endif // BOUSSINESQ

}
