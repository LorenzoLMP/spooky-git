#include "define_types.hpp"
#include "timestepping.hpp"
#include "cufft_routines.hpp"
#include "spooky.hpp"
#include "hydro_mhd_advance.hpp"
#include "cublas_routines.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "parameters.hpp"
#include "fields.hpp"
#include "supervisor.hpp"
#include "rkl.hpp"
#include "physics.hpp"



cublasStatus_t stat;
// extern int threadsPerBlock;


void TimeStepping::hydro_mhd_advance() {
    NVTX3_FUNC_RANGE();

    std::shared_ptr<Fields> fields = supervisor_ptr->fields;
    std::shared_ptr<Parameters> param = supervisor_ptr->param;
    std::shared_ptr<Physics> phys = supervisor_ptr->phys;

    cudaEventRecord(supervisor_ptr->start_2);


    // int blocksPerGrid = (2 * ntotal_complex * fields->num_fields + threadsPerBlock - 1) / threadsPerBlock;
    stage_step = 0;
    current_step += 1;


    // rewrite the following function to compute dt
    // and also it transforms fields from complex to real and stores the real fields
    // into the first [num_fields] tmparray (memory block starts at d_all_tmparray)
    compute_dt();

    // now we do the supertimestepping?
    // if we want to do it later we need to transform fields to real explicitely
// #if defined(SUPERTIMESTEPPING) && defined( ANISOTROPIC_DIFFUSION)
//     rkl->compute_cycle_STS(fields, param, *this, phys);
// #endif

    RungeKutta3();
    

    // if we want to do supertimestepping now we need to transform fields to real explicitely
// #if defined(SUPERTIMESTEPPING) && defined( ANISOTROPIC_DIFFUSION)
#if defined(SUPERTIMESTEPPING)
    Complex2RealFields(fields->d_all_fields, (scalar_type*)fields->d_all_tmparray, fields->num_fields);
    // // assign fields to [num_fields] tmparray (memory block starts at d_all_tmparray)
    // blocksPerGrid = ( fields->num_fields * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    // ComplexVecAssign<<<blocksPerGrid, threadsPerBlock>>>((data_type *)fields->d_all_fields, (data_type *)fields->d_all_tmparray, fields->num_fields * ntotal_complex);
    //
    // // compute FFTs from complex to real fields to start computation of shear traceless matrix
    // for (int n = 0; n < fields->num_fields; n++){
    //     c2r_fft(fields->d_tmparray[n], fields->d_tmparray_r[n], supervisor);
    // }
#if !defined(RKL)
    rkl->compute_cycle_STS();
#else
    rkl->compute_cycle_RKL();
#endif

#endif // supertimestepping

    cudaEventRecord(supervisor_ptr->stop_2);
    cudaEventSynchronize(supervisor_ptr->stop_2);
    supervisor_ptr->updateMainLooptime();

    return ;


}



const double gammaRK[3] = {8.0 / 15.0 , 5.0 / 12.0 , 3.0 / 4.0};
const double xiRK[2] = {-17.0 / 60.0 , -5.0 / 12.0};

void TimeStepping::RungeKutta3() {

#ifdef DDEBUG
    std::printf("Now entering RungeKutta3 function \n");
#endif

    std::shared_ptr<Fields> fields = supervisor_ptr->fields;
    std::shared_ptr<Parameters> param = supervisor_ptr->param;

    double dt_RK = 0.0;
    int blocksPerGrid = (2 * ntotal_complex * fields->num_fields + threadsPerBlock - 1) / threadsPerBlock;



    dt_RK = current_dt; // in theory one can do strang splitting so dt_RK can be 1/2 dt

#ifdef DDEBUG
    // std::printf("RK, finished 1st step.\n");
    // std::printf("After compute dfield, RK, 1st step:\n");
    // print_device_values();
    if (current_step == 1 || current_step % 100 == 0 ) std::printf("t: %.5e \t dt: %.5e \n",current_time,dt_RK);
    if (current_step == 1 || current_step % 100 == 0 ) fields->print_device_values();
#endif


#ifdef DDEBUG
    // std::printf("After 1st RK:\n");
    // print_device_values();
    // std::printf("num_fields : %d \n",fields->num_fields);
    std::printf("RK3, doing step n. %d ...\n",stage_step+1);
#endif

    compute_dfield();


    // d_all_fields = d_all_fields + gammaRK[0] * dt * d_all_dfields;
    axpyDouble<<<blocksPerGrid, threadsPerBlock>>>( (scalar_type *)fields->d_all_fields, (scalar_type *)fields->d_all_dfields, (scalar_type *)fields->d_all_fields, (scalar_type) 1.0, gammaRK[0]*dt_RK,  2 * ntotal_complex * fields->num_fields);
    // // d_all_scrtimestep = d_all_fields + xiRK[0] * dt * d_all_dfields;
    axpyDouble<<<blocksPerGrid, threadsPerBlock>>>( (scalar_type *)fields->d_all_fields, (scalar_type *)fields->d_all_dfields, (scalar_type *)d_all_scrtimestep, (scalar_type) 1.0, xiRK[0]*dt_RK,  2 * ntotal_complex * fields->num_fields);

    // end of stage 1
    stage_step++;

#ifdef DDEBUG
    std::printf("RK3, doing step n. %d ...\n",stage_step+1);
#endif
    // std::printf("...Computing dfield\n");
    compute_dfield();

    // for( i = 0 ; i < NTOTAL_COMPLEX ; i++) {
    //         fld.farray[n][i] = fld1.farray[n][i] + gammaRK[1] * dfld.farray[n][i] * dt;
    //         fld1.farray[n][i] = fld.farray[n][i] + xiRK[1] * dfld.farray[n][i] * dt;
    //     }

    // d_all_fields = d_all_scrtimestep + gammaRK[1] * dt * d_all_dfields;
    axpyDouble<<<blocksPerGrid, threadsPerBlock>>>( (scalar_type *)d_all_scrtimestep, (scalar_type *)fields->d_all_dfields, (scalar_type *)fields->d_all_fields, (scalar_type) 1.0, gammaRK[1]*dt_RK,  2 * ntotal_complex * fields->num_fields);
    // d_all_scrtimestep = d_all_fields + xiRK[1] * dt * d_all_dfields;
    axpyDouble<<<blocksPerGrid, threadsPerBlock>>>( (scalar_type *)fields->d_all_fields, (scalar_type *)fields->d_all_dfields, (scalar_type *)d_all_scrtimestep, (scalar_type) 1.0, xiRK[1]*dt_RK,  2 * ntotal_complex * fields->num_fields);

    // end of stage 2
    stage_step++;

#ifdef DDEBUG
    std::printf("RK3, doing step n. %d ...\n",stage_step+1);
#endif
    // std::printf("...Computing dfield\n");
    compute_dfield();

    // for( i = 0 ; i < NTOTAL_COMPLEX ; i++) {
    //     fld.farray[n][i] = fld1.farray[n][i] + gammaRK[2] * dfld.farray[n][i] * dt;
    // }
    // d_all_fields = d_all_scrtimestep + gammaRK[2] * dt * d_all_dfields;
    axpyDouble<<<blocksPerGrid, threadsPerBlock>>>( (scalar_type *)d_all_scrtimestep, (scalar_type *)fields->d_all_dfields, (scalar_type *)fields->d_all_fields, (scalar_type) 1.0, gammaRK[2]*dt_RK,  2 * ntotal_complex * fields->num_fields);

    current_time += current_dt;

    // end of stage 3
    stage_step++;
#ifdef DDEBUG
    std::printf("End of RK3 integrator, t: %.5e \t dt: %.5e \n",current_time,current_dt);
#endif


}

