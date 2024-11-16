#include "define_types.hpp"
#include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "timestepping.hpp"
#include "parameters.hpp"
#include "fields.hpp"
#include "physics.hpp"
#include "supervisor.hpp"

void TimeStepping::compute_dfield() {
    NVTX3_FUNC_RANGE();

    std::shared_ptr<Fields> fields = supervisor->fields;
    std::shared_ptr<Parameters> param = supervisor->param;
    std::shared_ptr<Physics> phys = supervisor->phys;

    int blocksPerGrid;
    /*
     * Do all computations
     * required to compute dfield
     *
     */
#ifdef DDEBUG
    std::printf("Now entering compute_dfield function \n");
#endif



#ifndef HEAT_EQ
    // for heat eq we do not need to compute ffts
    // complex to real because we can just use complex variables

    // assign fields to [num_fields] tmparray (memory block starts at d_all_tmparray)
    blocksPerGrid = ( fields->num_fields * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    ComplexVecAssign<<<blocksPerGrid, threadsPerBlock>>>((data_type *)fields->d_all_fields, (data_type *)fields->d_all_tmparray, fields->num_fields * ntotal_complex);

    // compute FFTs from complex to real fields to start computation of shear traceless matrix
    for (int n = 0; n < fields->num_fields; n++){
        c2r_fft(fields->d_tmparray[n], fields->d_tmparray_r[n], supervisor);
    }

#endif

    // cudaDeviceSynchronize();
    if (stage_step == 0) compute_dt();

#ifdef INCOMPRESSIBLE


    // compute hyperbolic terms
    phys->HyperbolicTerms();

#if defined(BOUSSINESQ) && defined(STRATIFICATION)
    // add - th e_strat to velocity component in the strat direction
    // add N2 u_strat to temperature equation
    // this is for normalization where theta is in units of g [L/T^2]
    // other normalizations possible
    phys->EntropyStratification();
#endif

/*
 *
 * Now we enforce the incompressibility
 * condition
 *
 */

    // compute pseudo-pressure and subtract grad p_tilde from dfields
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    GradPseudoPressure<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)fields->wavevector.d_all_kvec, (data_type *) fields->d_all_dfields, ntotal_complex);



#endif //end INCOMPRESSIBLE


#ifndef SUPERTIMESTEPPING
    // compute parabolic terms
    phys->ParabolicTerms(fields->d_all_fields, fields->d_all_dfields);
#endif

}




