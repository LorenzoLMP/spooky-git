#include "define_types.hpp"
#include "fields.hpp"
#include "cufft_routines.hpp"
#include "spooky.hpp"
#include "hydro_mhd_advance.hpp"
#include "cublas_routines.hpp"
// #include "compute_timestep.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"


const double gammaRK[3] = {8.0 / 15.0 , 5.0 / 12.0 , 3.0 / 4.0};
const double xiRK[2] = {-17.0 / 60.0 , -5.0 / 12.0};
cublasStatus_t stat;
// extern int threadsPerBlock;


void Fields::RungeKutta3( double t, double t_end, Parameters  *param) {
    NVTX3_FUNC_RANGE();
    double dt_RK = 0.0;
    int blocksPerGrid = (2 * ntotal_complex * num_fields + threadsPerBlock - 1) / threadsPerBlock;
    int RK_step = 0;
    current_step += 1;

    // compute_dt( );

    // std::printf("...Computing dfield\n");
    compute_dfield(RK_step++, param);
    if ( t + current_dt > t_end) current_dt = t_end - t;
    dt_RK = current_dt; // in theory one can do strang splitting so dt_RK can be 1/2 dt
#ifdef DEBUG
    std::printf("RK, 1st step:\n");
    std::printf("After compute dfield, RK, 1st step:\n");
    // print_device_values();
    if (current_step == 1 || current_step % 100 == 0 ) std::printf("t: %.5e \t dt: %.5e \n",t,dt_RK);
    if (current_step == 1 || current_step % 100 == 0 ) print_device_values();
#endif



    // snooopy code
    // for( n = 0 ; n < fld.nfield ; n++) {
    //     for( i = 0 ; i < NTOTAL_COMPLEX ; i++) {
    //         fld.farray[n][i] = fld.farray[n][i] + gammaRK[0] * dfld.farray[n][i] * dt;
    //         fld1.farray[n][i] = fld.farray[n][i] + xiRK[0] * dfld.farray[n][i] * dt;
    //     }
    // }

    // d_all_fields = d_all_fields + gammaRK[0] * dt * d_all_dfields;
    axpyDouble<<<blocksPerGrid, threadsPerBlock>>>( (scalar_type *)d_all_fields, (scalar_type *)d_all_dfields, (scalar_type *)d_all_fields, (scalar_type) 1.0, gammaRK[0]*dt_RK,  2 * ntotal_complex * num_fields);
    // // d_all_scrtimestep = d_all_fields + xiRK[0] * dt * d_all_dfields;
    axpyDouble<<<blocksPerGrid, threadsPerBlock>>>( (scalar_type *)d_all_fields, (scalar_type *)d_all_dfields, (scalar_type *)d_all_scrtimestep, (scalar_type) 1.0, xiRK[0]*dt_RK,  2 * ntotal_complex * num_fields);

#ifdef DEBUG
    std::printf("After 1st RK:\n");
    // print_device_values();
    std::printf("RK, 2nd step:\n");
#endif
    // std::printf("...Computing dfield\n");
    compute_dfield(RK_step++, param);
    // for( i = 0 ; i < NTOTAL_COMPLEX ; i++) {
    //         fld.farray[n][i] = fld1.farray[n][i] + gammaRK[1] * dfld.farray[n][i] * dt;
    //         fld1.farray[n][i] = fld.farray[n][i] + xiRK[1] * dfld.farray[n][i] * dt;
    //     }

    // d_all_fields = d_all_scrtimestep + gammaRK[1] * dt * d_all_dfields;
    axpyDouble<<<blocksPerGrid, threadsPerBlock>>>( (scalar_type *)d_all_scrtimestep, (scalar_type *)d_all_dfields, (scalar_type *)d_all_fields, (scalar_type) 1.0, gammaRK[1]*dt_RK,  2 * ntotal_complex * num_fields);
    // d_all_scrtimestep = d_all_fields + xiRK[1] * dt * d_all_dfields;
    axpyDouble<<<blocksPerGrid, threadsPerBlock>>>( (scalar_type *)d_all_fields, (scalar_type *)d_all_dfields, (scalar_type *)d_all_scrtimestep, (scalar_type) 1.0, xiRK[1]*dt_RK,  2 * ntotal_complex * num_fields);

#ifdef DEBUG
    std::printf("After 2nd RK:\n");
    // print_device_values();
    std::printf("RK, 3rd step:\n");
#endif
    // std::printf("...Computing dfield\n");
    compute_dfield(RK_step++, param);
    // for( i = 0 ; i < NTOTAL_COMPLEX ; i++) {
    //     fld.farray[n][i] = fld1.farray[n][i] + gammaRK[2] * dfld.farray[n][i] * dt;
    // }
    // d_all_fields = d_all_scrtimestep + gammaRK[2] * dt * d_all_dfields;
    axpyDouble<<<blocksPerGrid, threadsPerBlock>>>( (scalar_type *)d_all_scrtimestep, (scalar_type *)d_all_dfields, (scalar_type *)d_all_fields, (scalar_type) 1.0, gammaRK[2]*dt_RK,  2 * ntotal_complex * num_fields);


    return ;


}




// double Fields::advance_timestep( double t, double t_end, int* p_step) {
//     NVTX3_FUNC_RANGE();
//     double dt = 0.0;
//     // int step = 0;
//
//     // std::printf("Computing dt\n");
//     compute_dt( &dt );
//
//     if ( t + dt > t_end) dt = t_end - t;
//     // std::printf("dt: %.2e \n",dt);
//     // one can implement different solvers...
//     // RungeKutta3((double)0.5);
//     RungeKutta3((double)dt);
//     // std::printf("Updating n. step\n");
//     *p_step = *p_step + 1;
//     return dt;
// }
