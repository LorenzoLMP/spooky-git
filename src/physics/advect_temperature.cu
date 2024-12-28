#include "common.hpp"
#include "physics.hpp"
#include "fields.hpp"
#include "cufft_routines.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "parameters.hpp"
#include "supervisor.hpp"

void Physics::AdvectTemperature(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dFields) {

    int blocksPerGrid;
    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;
    scalar_type* mask = fields_ptr->wavevector.d_mask;


    if (param_ptr->boussinesq) {

        data_type* en_flux = fields_ptr->d_all_tmparray;

        scalar_type* real_velField = real_Buffer + vars.VEL * 2 * grid.NTOTAL_COMPLEX ;
        scalar_type* real_Theta = real_Buffer + vars.TH * 2 * grid.NTOTAL_COMPLEX ;

        // first compute energy flux vector [ u_x theta, u_y theta, u_z theta]
        // we can re-utilize tmparrays store result in in the temp_arrays from [0, 1, 2]
        blocksPerGrid = ( 2 * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        EnergyFluxVector<<<blocksPerGrid, threadsPerBlock>>>(real_velField, real_Theta, (scalar_type *) en_flux,  2 * grid.NTOTAL_COMPLEX);


        // take fourier transforms of the 3 energy flux vector components
        for (int n = 0; n < 3; n++) {
            r2c_fft(en_flux + 2*n*grid.NTOTAL_COMPLEX,  en_flux + n*grid.NTOTAL_COMPLEX, supervisor_ptr);
        }


        // compute derivative of energy flux vector and assign u nabla theta to the dfield for theta
        data_type* complex_dTheta = complex_dFields + vars.TH * grid.NTOTAL_COMPLEX ;

        blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        NonLinBoussinesqAdv<<<blocksPerGrid, threadsPerBlock>>>(kvec, en_flux, complex_dTheta, mask, grid.NTOTAL_COMPLEX);
    }
}
