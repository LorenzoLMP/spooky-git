#include "common.hpp"
#include "physics.hpp"
#include "fields.hpp"
#include "cufft_routines.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "parameters.hpp"
#include "supervisor.hpp"

void Physics::AnisotropicConduction(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dTheta) {

    int blocksPerGrid;

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;
    scalar_type* mask = fields_ptr->wavevector.d_mask;

    data_type* complex_Theta = complex_Fields + vars.TH * grid.NTOTAL_COMPLEX ;

    // scalar_type* real_velField = real_Buffer + vars.VEL * 2 * grid.NTOTAL_COMPLEX ;
    scalar_type* real_magField = real_Buffer + vars.MAG * 2 * grid.NTOTAL_COMPLEX ;


    if (param_ptr->anisotropic_diffusion) {

        // compute gradient of theta and assign it to the first 3 scratch arrays [0, 1, 2]
        data_type* grad_theta = fields_ptr->d_all_tmparray;

        blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        Gradient<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_Theta, grad_theta, grid.NTOTAL_COMPLEX);
        // compute complex to real iFFTs
        for (int n = 0; n < 3; n++){
            c2r_fft(grad_theta + n*grid.NTOTAL_COMPLEX, ((scalar_type *) grad_theta) + 2*n*grid.NTOTAL_COMPLEX , supervisor_ptr);
        }


        // compute the scalar B grad theta (real space) and assign it to [3] scratch array
        // Bx, By, Bz real fields are already in the 4-5-6 real_Buffer arrays
        // NOTE: for supertimestepping strictly speaking this is not 100% self-consistent because the real magnetic field in d_tmparray_r refer to those prior to the supertimestepping, so in theory before proceeding they would  need to be computed again from fields_in. However, with an isotropic magnetic resistivity, the value of b_x, b_y, b_z do not change during supertimestepping (even though B_x, B_y and B_z do). So here we are good, because \bm b is all that matters for the anisotropic conduction.


        scalar_type* Bgrad_theta = fields_ptr->d_tmparray_r[3];

        blocksPerGrid = ( 2 * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        ComputeBGradTheta<<<blocksPerGrid, threadsPerBlock>>>(real_magField, (scalar_type *) grad_theta, Bgrad_theta, 2 * grid.NTOTAL_COMPLEX);

        // compute the anisotropic heat flux and put it in the [0, 1, 2] tmp array
        scalar_type* heat_flux = fields_ptr->d_tmparray_r[0];

        ComputeAnisotropicHeatFlux<<<blocksPerGrid, threadsPerBlock>>>( real_magField, Bgrad_theta, heat_flux, param_ptr->OmegaT2, (1./param_ptr->reynolds_ani), 2 * grid.NTOTAL_COMPLEX, param_ptr->strat_direction);

        // take fourier transforms of the heat flux
        for (int n = 0 ; n < 3; n++) {
            r2c_fft(heat_flux + 2*n*grid.NTOTAL_COMPLEX, ((data_type*) heat_flux) + n*grid.NTOTAL_COMPLEX, supervisor_ptr);
        }

        // take divergence of heat flux and add to dtemp
        blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        DivergenceMask<<<blocksPerGrid, threadsPerBlock>>>(kvec, (data_type*) heat_flux, complex_dTheta, mask, grid.NTOTAL_COMPLEX, 1);
    }

}
