#include "define_types.hpp"
#include "fields.hpp"
#include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"


void Fields::EntropyStratification() {

    int blocksPerGrid;

#ifdef BOUSSINESQ
#ifdef STRATIFICATION
    // add - th e_strat to velocity component in the strat direction
    // add N2 u_strat to temperature equation
    // this is for normalization where theta is in units of g [L/T^2]
    // other normalizations possible
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    BoussinesqStrat<<<blocksPerGrid, threadsPerBlock>>>( (data_type *)d_all_fields, (data_type *) d_all_dfields, param->N2, ntotal_complex, STRAT_DIR);
#endif // STRATIFICATION
#endif // BOUSSINESQ

}
