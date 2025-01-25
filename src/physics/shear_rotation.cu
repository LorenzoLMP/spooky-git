#include "common.hpp"
#include "physics.hpp"
#include "fields.hpp"
#include "cufft_routines.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "parameters.hpp"
#include "supervisor.hpp"


void Physics::BackgroundShear(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dFields) {

    int blocksPerGrid;
    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    if (param_ptr->shearing) {

        // add du_y += param_ptr->shear * u_x
        // and dB_y -= param_ptr->shear * B_x (if MHD)
        // note the sign difference when the kernel is called

        data_type* complex_Velx = complex_Fields + vars.VX * grid.NTOTAL_COMPLEX ;
        data_type* complex_dVely = complex_dFields + vars.VY * grid.NTOTAL_COMPLEX ;

        blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        ShearingFlow<<<blocksPerGrid, threadsPerBlock>>>( complex_Velx, complex_dVely, param_ptr->shear, grid.NTOTAL_COMPLEX);

        if (param_ptr->mhd) {

            data_type* complex_Magx = complex_Fields + vars.BX * grid.NTOTAL_COMPLEX ;
            data_type* complex_dMagy = complex_dFields + vars.BY * grid.NTOTAL_COMPLEX ;

            ShearingFlow<<<blocksPerGrid, threadsPerBlock>>>( complex_Magx, complex_dMagy, - param_ptr->shear, grid.NTOTAL_COMPLEX);
        }
    }

}


void Physics::BackgroundRotation(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dFields) {

    int blocksPerGrid;
    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    if (param_ptr->shearing) {

        // add du_x += 2.0 * param_ptr->omega * u_y
        // add du_y -= 2.0 * param_ptr->omega * u_x

        data_type* complex_Velx = complex_Fields + vars.VX * grid.NTOTAL_COMPLEX ;
        data_type* complex_Vely = complex_Fields + vars.VY * grid.NTOTAL_COMPLEX ;
        data_type* complex_dVelx = complex_dFields + vars.VX * grid.NTOTAL_COMPLEX ;
        data_type* complex_dVely = complex_dFields + vars.VY * grid.NTOTAL_COMPLEX ;

        blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        CoriolisForce<<<blocksPerGrid, threadsPerBlock>>>( complex_Velx, complex_Vely, complex_dVelx, complex_dVely, param_ptr->omega, grid.NTOTAL_COMPLEX);


    }

}



