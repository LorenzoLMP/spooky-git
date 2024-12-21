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

void TimeStepping::compute_dfield(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dFields) {
    NVTX3_FUNC_RANGE();

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;
    std::shared_ptr<Physics> phys_ptr = supervisor_ptr->phys_ptr;

    int blocksPerGrid;

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;
    // scalar_type* mask = fields_ptr->wavevector.d_mask;
    /*
     * Do all computations
     * required to compute dfield
     *
     */

    if (param_ptr->debug > 0) {
        std::printf("Now entering compute_dfield function \n");
    }



    // for heat eq we do not need to compute ffts
    // complex to real because we can just use complex variables

    if (param_ptr->incompressible) {

        // before we do anything we need to transform from
        // complex to real. However, when stage_step == 0
        // (at the beginning of the hydro_mhd_advance function)
        // this has already been done by the compute_dt function

        if (stage_step > 0) {
            // Complex2RealFields(fields_ptr->d_all_fields, fields_ptr->d_all_buffer_r) which
            // copies the complex fields from d_all_fields into d_all_buffer_r and performs
            // an in-place r2c FFT to give the real fields. This buffer is reserved for the real fields!

            supervisor_ptr->Complex2RealFields(complex_Fields, real_Buffer, fields_ptr->num_fields);
        }

        // compute hyperbolic terms
        phys_ptr->HyperbolicTerms(complex_Fields, real_Buffer, complex_dFields);

        if (param_ptr->stratification) {
            // add - th e_strat to velocity component in the strat direction
            // add N2 u_strat to temperature equation
            // this is for normalization where theta is in units of g [L/T^2]
            // other normalizations possible
            phys_ptr->EntropyStratification(complex_Fields, real_Buffer, complex_dFields);
        }

        /*
        *
        * Now we enforce the incompressibility
        * condition
        *
        */

        // compute pseudo-pressure and subtract grad p_tilde from dfields
        blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
        GradPseudoPressure<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_dFields, ntotal_complex);

    } //end INCOMPRESSIBLE

    if (not param_ptr->supertimestepping) {
        // compute parabolic terms
        phys_ptr->ParabolicTerms(complex_Fields, real_Buffer, complex_dFields);
    }

}




