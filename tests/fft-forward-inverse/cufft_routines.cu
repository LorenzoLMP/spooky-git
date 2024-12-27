#include <cufftXt.h>
#include "common.hpp"
#include "cufft_routines.hpp"
#include "cuda_kernels_generic.hpp"
// #include "supervisor.hpp"
// cufftHandle plan_r2c{}, plan_c2r{};

cufftHandle plan_r2c{};
cufftHandle plan_c2r{};

// extern const int threadsPerBlock;

void r2c_fft(void *r_data_in, void *c_data_out) {

    // Execute the plan_r2c
    CUFFT_CALL(cufftXtExec(plan_r2c, r_data_in, c_data_out, CUFFT_FORWARD));


};


void c2r_fft(void *c_data_in, void *r_data_out) {

    // Scale complex results
    int dimGrid, dimBlock;
    dimGrid = (grid.NTOTAL + threadsPerBlock - 1) / threadsPerBlock;
    dimBlock = threadsPerBlock;

    scaleKernel<<<dimGrid, dimBlock>>>(reinterpret_cast<cufftDoubleComplex *>(c_data_in), (double) 1./(grid.FFT_SIZE[0] * grid.FFT_SIZE[1] * grid.FFT_SIZE[2]), grid.FFT_SIZE[0] * grid.FFT_SIZE[1] * ((grid.FFT_SIZE[2] / 2) + 1));
    //CUDA_RT_CALL( cudaPeekAtLastError() );
    //CUDA_RT_CALL( cudaDeviceSynchronize() );

    // Execute the plan_c2r
    CUFFT_CALL(cufftXtExec(plan_c2r, c_data_in, r_data_out, CUFFT_INVERSE));



};

void init_plan(const size_t *fft_size) {
    // Initiate cufft plans, one for r2c and one for c2r

    CUFFT_CALL(cufftCreate(&plan_r2c));
    CUFFT_CALL(cufftCreate(&plan_c2r));

    // Create the plans
    size_t workspace_size;
    CUFFT_CALL(cufftMakePlan3d(plan_r2c, fft_size[0], fft_size[1], fft_size[2], CUFFT_D2Z, &workspace_size));
    CUFFT_CALL(cufftMakePlan3d(plan_c2r, fft_size[0], fft_size[1], fft_size[2], CUFFT_Z2D, &workspace_size));


};

void finish_cufft() {
    CUFFT_CALL(cufftDestroy(plan_r2c));
    CUFFT_CALL(cufftDestroy(plan_c2r));
};
