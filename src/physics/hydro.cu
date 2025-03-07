#include "common.hpp"
#include "physics.hpp"
// #include "timestepping.hpp"
// #include "cufft_routines.hpp"
#include "cublas_routines.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "parameters.hpp"
// #include "inputoutput.hpp"
#include "fields.hpp"
#include <cuda_runtime.h>
// #include <cufftXt.h>
#include "cufft_utils.h"
#include "cufft_routines.hpp"
#include "supervisor.hpp"
// #include "timestepping.hpp"
// #include "rkl.hpp"

void Physics::BasdevantHydro(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dVel) {

    /*
    *
    * Here we do the Basdevant hydro nonlinear terms
    *
    */

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    int blocksPerGrid;

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;
    scalar_type* mask = fields_ptr->wavevector.d_mask;

    scalar_type* real_velField = real_Buffer + vars.VEL * 2 * grid.NTOTAL_COMPLEX ;
    // data_type* complex_dVel = complex_dFields + vars.VEL * grid.NTOTAL_COMPLEX ;


    if (param_ptr->incompressible) {

        // we use Basdevant formulation [1983]
        // compute the elements of the traceless symmetric matrix
        // B_ij = u_i u_j - delta_ij Tr (u_i u_j) / 3.
        // It has only 5 independent components B_xx, B_xy, B_xz, Byy, B_yz.
        // (B_zz = - B_xx - B_yy)
        // The results are saved in the temp_arrays from [0, 1, ..., 4]
        data_type* shear_matrix = fields_ptr->d_all_tmparray;

        blocksPerGrid = ( 2 * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;

        if (param_ptr->mhd) {

            scalar_type* real_magField = real_Buffer + vars.MAG * 2 * grid.NTOTAL_COMPLEX ;
            // data_type* complex_dMag = complex_dFields + vars.MAG * grid.NTOTAL_COMPLEX ;

            TracelessShearMatrixMHD<<<blocksPerGrid, threadsPerBlock>>>(real_velField, real_magField, (scalar_type*) shear_matrix,  2 * grid.NTOTAL_COMPLEX);
        }
        else {
            TracelessShearMatrix<<<blocksPerGrid, threadsPerBlock>>>(real_velField, (scalar_type*) shear_matrix,  2 * grid.NTOTAL_COMPLEX);
        }


        // take fft of 5 independent components of B_ij
        for (int n = 0; n < 5; n++) {
            r2c_fft((scalar_type*) shear_matrix + 2*n*grid.NTOTAL_COMPLEX, shear_matrix + n*grid.NTOTAL_COMPLEX, supervisor_ptr);
        }

        // compute derivative of traceless shear matrix and assign to dfields
        // this kernel works also if MHD
        blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        NonLinHydroAdv<<<blocksPerGrid, threadsPerBlock>>>(kvec, shear_matrix, complex_dVel, mask, grid.NTOTAL_COMPLEX);

    }

}

// This is a convenience function, it is used for the outputs
void Physics::NonLinearAdvection(scalar_type* real_vecField, data_type* advectionVec) {

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    int blocksPerGrid;

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;
    scalar_type* mask = fields_ptr->wavevector.d_mask;

    // compute the elements of the shear symmetric matrix
    // S_ij = u_i u_j 
    // It has 6 independent components.
    // The results are saved in the temp_arrays from [0, 1, ..., 5]
    data_type* shear_matrix = fields_ptr->d_all_tmparray;
    
    blocksPerGrid = ( 2 * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
    ShearMatrix<<<blocksPerGrid, threadsPerBlock>>>(real_vecField, (scalar_type*) shear_matrix,  2 * grid.NTOTAL_COMPLEX);

    // take fft of 6 independent components of S_ij
    for (int n = 0; n < 6; n++) {
        r2c_fft((scalar_type*) shear_matrix + 2*n*grid.NTOTAL_COMPLEX, shear_matrix + n*grid.NTOTAL_COMPLEX, supervisor_ptr);
    }

    // compute derivative of shear matrix and assign it to advectionVec
    blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
    NonLinAdvection<<<blocksPerGrid, threadsPerBlock>>>(kvec, shear_matrix, advectionVec, mask, grid.NTOTAL_COMPLEX);

}