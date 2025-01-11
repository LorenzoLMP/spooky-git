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


Physics::Physics(Supervisor &sup_in){

    supervisor_ptr = &sup_in;

}

Physics::~Physics(){

}

void Physics::HyperbolicTerms(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dFields){
    /*
    *
    * Here we do the hyperbolic terms
    *
    */

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    if (param_ptr->debug > 1) {
        std::printf("Now entering HyperbolicTerms function \n");
    }

    int blocksPerGrid;

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;
    scalar_type* mask = fields_ptr->wavevector.d_mask;

    if (param_ptr->incompressible) {

        scalar_type* real_velField = real_Buffer + vars.VEL * 2 * grid.NTOTAL_COMPLEX ;
        data_type* complex_dVel = complex_dFields + vars.VEL * grid.NTOTAL_COMPLEX ;

        scalar_type* real_magField = real_Buffer + vars.MAG * 2 * grid.NTOTAL_COMPLEX ;
        data_type* complex_dMag = complex_dFields + vars.MAG * grid.NTOTAL_COMPLEX ;
//         if (param_ptr->mhd) {
//
//
//         }

        // we use Basdevant formulation [1983]
        // compute the elements of the traceless symmetric matrix
        // B_ij = u_i u_j - delta_ij Tr (u_i u_j) / 3.
        // It has only 5 independent components B_xx, B_xy, B_xz, Byy, B_yz.
        // (B_zz = - B_xx - B_yy)
        // The results are saved in the temp_arrays from [0, 1, ..., 4]
        data_type* shear_matrix = fields_ptr->d_all_tmparray;

        blocksPerGrid = ( 2 * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;

        if (param_ptr->mhd) {

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

        if (param_ptr->mhd) {

            // compute emf = u x B:
            // emf_x = u_y B_z - u_z B_y , emf_y = u_z B_x - u_x B_z , emf_z = u_x B_y - u_y B_x
            // the results are saved in the first 3 temp_arrays as [emf_x, emf_y, emf_z] (they are the x,y,z components of the emf)
            // We can re-utilize tmparrays and store result in in the temp_arrays from [0, 1, 2]

            data_type* emf = fields_ptr->d_all_tmparray;

            blocksPerGrid = ( 2 * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
            MagneticEmf<<<blocksPerGrid, threadsPerBlock>>>(real_velField, real_magField, (scalar_type*) emf,  2 * grid.NTOTAL_COMPLEX);

            // take fourier transforms of the 3 independent components of the antisymmetric shear matrix
            for (int n = 0; n < 3; n++) {
                r2c_fft((scalar_type*) emf + 2*n*grid.NTOTAL_COMPLEX, emf + n*grid.NTOTAL_COMPLEX, supervisor_ptr);
            }

            // compute derivative of antisymmetric magnetic shear matrix and assign to dfields

            blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
            MagneticShear<<<blocksPerGrid, threadsPerBlock>>>(kvec, emf, complex_dMag, mask, grid.NTOTAL_COMPLEX);

        }

        if (param_ptr->boussinesq) {
            // does the advection of the temperature
            AdvectTemperature(complex_Fields, real_Buffer, complex_dFields);
        }

    }
}


void Physics::ParabolicTerms(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dFields){
    /*
    *
    * Here we do the diffusion terms
    *
    */


    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    if (param_ptr->debug > 1) {
        std::printf("Now entering ParabolicTerms function \n");
    }

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;

    int blocksPerGrid;

    blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;


    if (param_ptr->incompressible) {
        // for explicit treatment of diffusion terms
        // with incompressible d_all_fields always points at vars.VX

        data_type* complex_velField = complex_Fields + vars.VEL * grid.NTOTAL_COMPLEX ;
        data_type* complex_dVel = complex_dFields + vars.VEL * grid.NTOTAL_COMPLEX ;

        nablaOpVector<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_velField, complex_dVel, param_ptr->nu, (size_t) grid.NTOTAL_COMPLEX, 1);
    }

    if (param_ptr->mhd) {
        // for explicit treatment of diffusion terms
        // point d_all_fields at vars.BX

        data_type* complex_magField = complex_Fields + vars.MAG * grid.NTOTAL_COMPLEX ;
        data_type* complex_dMag = complex_dFields + vars.MAG * grid.NTOTAL_COMPLEX ;

        nablaOpVector<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_magField, complex_dMag, param_ptr->nu_m, (size_t) grid.NTOTAL_COMPLEX, 1);
    }

    if (param_ptr->boussinesq or param_ptr->heat_equation) {

        data_type* complex_Theta = complex_Fields + vars.TH * grid.NTOTAL_COMPLEX ;
        data_type* complex_dTheta = complex_dFields + vars.TH * grid.NTOTAL_COMPLEX ;

        if (param_ptr->anisotropic_diffusion) {
            AnisotropicConduction(complex_Fields, real_Buffer, complex_dTheta);
        }
        else {
            if (param_ptr->heat_equation) {
                // this is because the nabla scalar will *add* to d_dfarray, and with HEAT_EQ we want to *set*
                VecInitComplex<<<blocksPerGrid, threadsPerBlock>>>(complex_dTheta, data_type(0.0,0.0), grid.NTOTAL_COMPLEX);
            }
            nablaOpScalar<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_Theta, complex_dTheta, param_ptr->nu_th, (size_t) grid.NTOTAL_COMPLEX, 1);
        }
    }

}
