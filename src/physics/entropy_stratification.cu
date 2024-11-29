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

void Physics::EntropyStratification(data_type *fields_in, data_type *dfields_out) {

    int blocksPerGrid;
    std::shared_ptr<Fields> fields = supervisor_ptr->fields;
    std::shared_ptr<Parameters> param = supervisor_ptr->param;

#ifdef BOUSSINESQ
#ifdef STRATIFICATION
    // add - th e_strat to velocity component in the strat direction
    // add N2 u_strat to temperature equation
    // this is for normalization where theta is in units of g [L/T^2]
    // other normalizations possible
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    BoussinesqStrat<<<blocksPerGrid, threadsPerBlock>>>( fields_in, dfields_out, param->N2, ntotal_complex, STRAT_DIR);
#endif // STRATIFICATION
#endif // BOUSSINESQ

}
