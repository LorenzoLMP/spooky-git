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
        blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        BoussinesqStrat<<<blocksPerGrid, threadsPerBlock>>>( complex_Fields, complex_dFields, param_ptr->N2, grid.NTOTAL_COMPLEX, STRAT_DIR);
    }
}
