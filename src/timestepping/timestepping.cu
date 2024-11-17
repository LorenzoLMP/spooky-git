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

TimeStepping::TimeStepping(int num, Parameters &param, Supervisor &sup) {
    // param = &p_in;
    // fields = &f_in;

    supervisor = &sup;
    rkl = new RKLegendre(num, param, sup);
    // std::printf("The TimeSpentInFFTs is: %.4e",supervisor->TimeSpentInFFTs);
    current_dt = 0.0;
    current_time = 0.0;
    current_step = 0;
    // stage_step = 0;

    // this is the mega array that contains intermediate fields during multi-stage timestepping
    // std::printf("num fields ts: %d \n", fields->num_fields);
    std::printf("num timestepping scratch arrays: %d \n",num);
    CUDA_RT_CALL(cudaMalloc(&d_all_scrtimestep, (size_t) sizeof(data_type) * ntotal_complex * num));
    int blocksPerGrid = ( 2 * num * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    VecInit<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)d_all_scrtimestep, 0.0, 2 * ntotal_complex * num);
}

TimeStepping::~TimeStepping(){
    CUDA_RT_CALL(cudaFree(d_all_scrtimestep));
    delete rkl;
}
