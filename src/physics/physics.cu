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
#include "timestepping.hpp"
#include "rkl.hpp"


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

    // int blocksPerGrid;

    // scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;
    // scalar_type* mask = fields_ptr->wavevector.d_mask;

    // data_type* complex_dVel = complex_dFields + vars.VEL * grid.NTOTAL_COMPLEX ;

    if (param_ptr->incompressible) {

        data_type* complex_dVel = complex_dFields + vars.VEL * grid.NTOTAL_COMPLEX ;

        BasdevantHydro(complex_Fields, real_Buffer, complex_dVel);

    //     // we use Basdevant formulation [1983]
    //     // compute the elements of the traceless symmetric matrix
    //     // B_ij = u_i u_j - delta_ij Tr (u_i u_j) / 3.
    //     // It has only 5 independent components B_xx, B_xy, B_xz, Byy, B_yz.
    //     // (B_zz = - B_xx - B_yy)
    //     // The results are saved in the temp_arrays from [0, 1, ..., 4]
    //     data_type* shear_matrix = fields_ptr->d_all_tmparray;

    //     blocksPerGrid = ( 2 * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;

    //     if (param_ptr->mhd) {

    //         TracelessShearMatrixMHD<<<blocksPerGrid, threadsPerBlock>>>(real_velField, real_magField, (scalar_type*) shear_matrix,  2 * grid.NTOTAL_COMPLEX);
    //     }
    //     else {
    //         TracelessShearMatrix<<<blocksPerGrid, threadsPerBlock>>>(real_velField, (scalar_type*) shear_matrix,  2 * grid.NTOTAL_COMPLEX);
    //     }


    //     // take fft of 5 independent components of B_ij
    //     for (int n = 0; n < 5; n++) {
    //         r2c_fft((scalar_type*) shear_matrix + 2*n*grid.NTOTAL_COMPLEX, shear_matrix + n*grid.NTOTAL_COMPLEX, supervisor_ptr);
    //     }

    //     // compute derivative of traceless shear matrix and assign to dfields
    //     // this kernel works also if MHD
    //     blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
    //     NonLinHydroAdv<<<blocksPerGrid, threadsPerBlock>>>(kvec, shear_matrix, complex_dVel, mask, grid.NTOTAL_COMPLEX);

    }

    if (param_ptr->mhd) {

        data_type* complex_dMag = complex_dFields + vars.MAG * grid.NTOTAL_COMPLEX ;

        CurlEMF(complex_Fields, real_Buffer, complex_dMag);

    }

    if (param_ptr->boussinesq) {

        data_type* complex_dTheta = complex_dFields + vars.TH * grid.NTOTAL_COMPLEX ;

        // does the advection of the temperature
        AdvectTemperature(complex_Fields, real_Buffer, complex_dTheta);
    }
}

void Physics::SourceTerms(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dFields){
    /*
    *
    * Here we do the source terms
    *
    */

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    if (param_ptr->debug > 1) {
        std::printf("Now entering SourceTerms function \n");
    }

    if (param_ptr->stratification) {
        // add - th e_strat to velocity component in the strat direction
        // add N2 u_strat to temperature equation
        // this is for normalization where theta is in units of g [L/T^2]
        // other normalizations possible
        // data_type* complex_dTheta = complex_dFields + vars.TH * grid.NTOTAL_COMPLEX ;

        EntropyStratification(complex_Fields, real_Buffer, complex_dFields);
    }

    if (param_ptr->shearing) {
        // add du_y += param_ptr->shear * u_x
        // and dB_y += param_ptr->shear * B_x (if MHD)
        BackgroundShear(complex_Fields, real_Buffer, complex_dFields);
    }

    if (param_ptr->rotating) {
        // add du_x += 2.0 * param_ptr->omega * u_y
        // add du_y -= 2.0 * param_ptr->omega * u_x
        BackgroundRotation(complex_Fields, real_Buffer, complex_dFields);
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
    std::vector<int> sts_variables_pos = supervisor_ptr->timestep_ptr->rkl_ptr->sts_variables_pos;

    if (param_ptr->debug > 1) {
        std::printf("Now entering ParabolicTerms function \n");
    }

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;

    int blocksPerGrid;

    blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;


    if (param_ptr->incompressible) {

        for (int i = 0; i < 3; i++) {

            // if ((sts_variables_bool[vars.VEL + i] + STS)%2+1) {}

            if (sts_variables_pos[vars.VEL + i] < 0) {
                // the variable is not in the sts list
                // and should be evolved during the
                // parabolic step

                data_type* complex_velField = complex_Fields + (vars.VEL + i) * grid.NTOTAL_COMPLEX ;
                data_type* complex_dVel = complex_dFields + (vars.VEL + i) * grid.NTOTAL_COMPLEX ;

                nablaOpScalar<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_velField, complex_dVel, param_ptr->nu, (size_t) grid.NTOTAL_COMPLEX, 1);
            }
        }
    }

    if (param_ptr->mhd) {

        for (int i = 0; i < 3; i++) {

            if (sts_variables_pos[vars.MAG + i] < 0) {
                // the variable is not in the sts list
                // and should be evolved during the
                // parabolic step

                data_type* complex_magField = complex_Fields + (vars.MAG + i) * grid.NTOTAL_COMPLEX ;
                data_type* complex_dMag = complex_dFields + (vars.MAG + i) * grid.NTOTAL_COMPLEX ;

                nablaOpScalar<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_magField, complex_dMag, param_ptr->nu_m, (size_t) grid.NTOTAL_COMPLEX, 1);
            }
        }
    }

    if (param_ptr->boussinesq or param_ptr->heat_equation) {
        if (sts_variables_pos[vars.TH] < 0) {
            // the variable is not in the sts list
            // and should be evolved during the
            // parabolic step

            data_type* complex_Theta = complex_Fields + vars.TH * grid.NTOTAL_COMPLEX ;
            data_type* complex_dTheta = complex_dFields + vars.TH * grid.NTOTAL_COMPLEX ;

            if (param_ptr->anisotropic_diffusion) {
                AnisotropicHeatFlux(complex_Fields, real_Buffer, complex_dTheta);
            }
            else {
                if (param_ptr->heat_equation) {
                    // this is because the nabla scalar will *add* to d_dfarray, and with HEAT_EQ we want to *set*
                    // VecInitComplex<<<blocksPerGrid, threadsPerBlock>>>(complex_dTheta, data_type(0.0,0.0), grid.NTOTAL_COMPLEX);
                    nablaOpScalar<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_Theta, complex_dTheta, param_ptr->nu_th, (size_t) grid.NTOTAL_COMPLEX, 0);
                }
                else {
                    nablaOpScalar<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_Theta, complex_dTheta, param_ptr->nu_th, (size_t) grid.NTOTAL_COMPLEX, 1);
                }

            }
        }
    }

}


void Physics::ParabolicTermsSTS(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dFields){
    /*
    *
    * Here we do the diffusion terms for STS timestep
    * When it is called with STS complex_dFields size is
    * sts_variables.size() <= vars.NUM_FIELDS
    * sts_variables_pos holds the index in sts_variables
    * where the sts var should go (or -1 if the var is not
    * evolved with sts)
    * E.g. for sts_variables ("th") with MHD+boussinesq
    * sts_variables_pos = ( -1, -1, -1, -1, -1, -1, 0)
    * E.g.2: for sts variables ("by", "bz", "bx") with MHD
    * sts_variables_pos = ( -1, -1, -1, 2, 0, 1)
    */


    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;
    std::vector<int> sts_variables_pos = supervisor_ptr->timestep_ptr->rkl_ptr->sts_variables_pos;

    if (param_ptr->debug > 1) {
        std::printf("Now entering ParabolicTerms function for STS step \n");
    }

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;

    int blocksPerGrid;

    blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;


    if (param_ptr->incompressible) {

        for (int i = 0; i < 3; i++) {

            // if ((sts_variables_bool[vars.VEL + i] + STS)%2+1) {}

            if (sts_variables_pos[vars.VEL + i] >= 0) {
                // the variable is in the sts list
                // and should be evolved during the
                // STS step

                data_type* complex_velField = complex_Fields + (vars.VEL + i) * grid.NTOTAL_COMPLEX ;
                data_type* complex_dVel = complex_dFields + sts_variables_pos[vars.VEL + i] * grid.NTOTAL_COMPLEX ;

                nablaOpScalar<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_velField, complex_dVel, param_ptr->nu, (size_t) grid.NTOTAL_COMPLEX, 1);
            }
        }
    }

    if (param_ptr->mhd) {

        for (int i = 0; i < 3; i++) {

            if (sts_variables_pos[vars.MAG + i] >= 0) {
                // the variable is not in the sts list
                // and should be evolved during the
                // parabolic step

                data_type* complex_magField = complex_Fields + (vars.MAG + i) * grid.NTOTAL_COMPLEX ;
                data_type* complex_dMag = complex_dFields + sts_variables_pos[vars.MAG + i] * grid.NTOTAL_COMPLEX ;

                nablaOpScalar<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_magField, complex_dMag, param_ptr->nu_m, (size_t) grid.NTOTAL_COMPLEX, 1);
            }
        }
    }

    if (param_ptr->boussinesq or param_ptr->heat_equation) {

        if (sts_variables_pos[vars.TH] >= 0) {
            // the variable is not in the sts list
            // and should be evolved during the
            // parabolic step

            data_type* complex_Theta = complex_Fields + vars.TH * grid.NTOTAL_COMPLEX ;
            data_type* complex_dTheta = complex_dFields + sts_variables_pos[vars.TH] * grid.NTOTAL_COMPLEX ;

            if (param_ptr->anisotropic_diffusion) {
                AnisotropicHeatFlux(complex_Fields, real_Buffer, complex_dTheta);
            }
            else {
                if (param_ptr->heat_equation) {
                    // this is because the nabla scalar will *add* to d_dfarray, and with HEAT_EQ we want to *set*
                    // VecInitComplex<<<blocksPerGrid, threadsPerBlock>>>(complex_dTheta, data_type(0.0,0.0), grid.NTOTAL_COMPLEX);
                    nablaOpScalar<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_Theta, complex_dTheta, param_ptr->nu_th, (size_t) grid.NTOTAL_COMPLEX, 0);
                }
                else {
                    nablaOpScalar<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_Theta, complex_dTheta, param_ptr->nu_th, (size_t) grid.NTOTAL_COMPLEX, 1);
                }

            }
        }
    }

}
