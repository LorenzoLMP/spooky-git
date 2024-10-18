#include "define_types.hpp"
#include "rkl.hpp"
// #include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "cublas_routines.hpp"
#include "cuda_kernels.hpp"
#include "parameters.hpp"
// #include "inputoutput.hpp"
#include "fields.hpp"
#include "physics.hpp"
#include <cuda_runtime.h>
// #include <cufftXt.h>
// #include "spooky.hpp"
#include "cufft_utils.h"
// #include "define_types.hpp"
#include "supervisor.hpp"
#include "timestepping.hpp"

RKLegendre::RKLegendre(int num, Parameters &param, Supervisor &sup) {
    // param = &p_in;
    // fields = &f_in;

    supervisor = &sup;
    // std::printf("The TimeSpentInFFTs is: %.4e",supervisor->TimeSpentInFFTs);
    dt = 0.0;
    stage = 0;
    cfl_rkl = param->cfl_par;
    rmax_par = param->safety_sts;
    // this is the mega array that contains intermediate fields during multi-stage timestepping
    // std::printf("num fields ts: %d \n", fields->num_fields);
    std::printf("num rkl scratch arrays: %d \n",4);


    CUDA_RT_CALL(cudaMalloc(&d_all_dU, (size_t) sizeof(data_type) * ntotal_complex));
    CUDA_RT_CALL(cudaMalloc(&d_all_dU0, (size_t) sizeof(data_type) * ntotal_complex));
    CUDA_RT_CALL(cudaMalloc(&d_all_Uc0, (size_t) sizeof(data_type) * ntotal_complex));
    CUDA_RT_CALL(cudaMalloc(&d_all_Uc1, (size_t) sizeof(data_type) * ntotal_complex));
}

RKLegendre::~RKLegendre(){
    CUDA_RT_CALL(cudaFree(d_all_dU));
    CUDA_RT_CALL(cudaFree(d_all_dU0));
    CUDA_RT_CALL(cudaFree(d_all_Uc0));
    CUDA_RT_CALL(cudaFree(d_all_Uc1));
}


void compute_cycle(Fields &fields, Timestepping &timestep, Physics &phys){

    double dt_hyp = timestep->current_dt;

}
