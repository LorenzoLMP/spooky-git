#include "common.hpp"
#include "physics.hpp"
#include "fields.hpp"
#include "cufft_routines.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "parameters.hpp"
#include "supervisor.hpp"

void Physics::EntropyStratification(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dFields) {

    int blocksPerGrid;
    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    if (param_ptr->stratification) {

        // add - th e_strat to velocity component in the strat direction
        // add N2 u_strat to temperature equation
        // this is for normalization where theta is in units of g [L/T^2]
        // other normalizations possible

        data_type* complex_velField = complex_Fields + vars.VEL * grid.NTOTAL_COMPLEX ;
        data_type* complex_dVel = complex_dFields + vars.VEL * grid.NTOTAL_COMPLEX ;

        data_type* complex_Theta = complex_Fields + vars.TH * grid.NTOTAL_COMPLEX ;
        data_type* complex_dTheta = complex_dFields + vars.TH * grid.NTOTAL_COMPLEX ;


        blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        BoussinesqStrat<<<blocksPerGrid, threadsPerBlock>>>( complex_velField, complex_Theta, complex_dVel, complex_dTheta, param_ptr->N2, grid.NTOTAL_COMPLEX, STRAT_DIR);
    }
}
