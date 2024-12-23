#include "define_types.hpp"
#include "timestepping.hpp"
// #include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "cublas_routines.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "parameters.hpp"
// #include "inputoutput.hpp"
#include "fields.hpp"
#include <cuda_runtime.h>
// #include <cufftXt.h>
// #include "spooky.hpp"
#include "cufft_utils.h"
// #include "define_types.hpp"
#include "supervisor.hpp"
#include "rkl.hpp"

TimeStepping::TimeStepping(Supervisor &sup_in, Parameters &p_in) {
    // param = &p_in;
    // fields = &f_in;

    supervisor_ptr = &sup_in;
    // rkl = new RKLegendre(vars.NUM_FIELDS, param, supervisor);
    rkl = std::unique_ptr<RKLegendre> (new RKLegendre(vars.NUM_FIELDS, p_in, sup_in));
    // std::printf("The TimeSpentInFFTs is: %.4e",supervisor_ptr->TimeSpentInFFTs);
    current_dt = 0.0;
    current_time = 0.0;
    current_step = 0;
    // stage_step = 0;

    // this is the mega array that contains intermediate fields during multi-stage timestepping
    // std::printf("vars.NUM_FIELDS fields ts: %d \n", fields->vars.NUM_FIELDS_fields);
    std::printf("num timestepping scratch arrays: %d \n",vars.NUM_FIELDS);
    CUDA_RT_CALL(cudaMalloc(&d_all_scrtimestep, (size_t) sizeof(data_type) * grid.NTOTAL_COMPLEX * vars.NUM_FIELDS));
    int blocksPerGrid = ( 2 * vars.NUM_FIELDS * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
    VecInit<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)d_all_scrtimestep, 0.0, 2 * grid.NTOTAL_COMPLEX * vars.NUM_FIELDS);
}

TimeStepping::~TimeStepping(){
    CUDA_RT_CALL(cudaFree(d_all_scrtimestep));
    // delete rkl;
}
