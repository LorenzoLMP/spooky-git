#include "common.hpp"
#include "timestepping.hpp"
#include "cufft_routines.hpp"
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


void TimeStepping::HydroMHDAdvance(std::shared_ptr<Fields> fields_ptr) {
    NVTX3_FUNC_RANGE();


    // std::shared_ptr<Fields> fields = supervisor_ptr->fields;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;
    // std::shared_ptr<Physics> phys_ptr = supervisor_ptr->phys;

    cudaEventRecord(supervisor_ptr->start_2);


    // int blocksPerGrid = (2 * grid.NTOTAL_COMPLEX * vars.NUM_FIELDS + threadsPerBlock - 1) / threadsPerBlock;
    stage_step = 0;
    current_step += 1;

    // the following function computes the timestep dt and it also calls internally
    // Complex2RealFields(fields_ptr->d_all_fields, fields_ptr->d_all_buffer_r) which
    // copies the complex fields from d_all_fields into d_all_buffer_r and performs
    // an in-place r2c FFT to give the real fields. This buffer is reserved for the real fields!
    // This is necessary to compute dt and is then used by the time-advancing algorithm
    // (e.g. RungeKutta3).
    compute_dt(fields_ptr->d_all_fields, fields_ptr->d_all_buffer_r);

    // now we do the supertimestepping?
    // if we want to do it later we need to transform fields to real explicitely
    // if (param_ptr->supertimestepping and (current_step%2)==1) {
    //
    //     rkl_ptr->compute_cycle(fields_ptr->d_all_fields, fields_ptr->d_all_buffer_r);
    // }


    // RungeKutta3(fields_ptr->d_all_fields, fields_ptr->d_all_buffer_r);
    // this is only for testing the super time stepping
    // in this case also deactivate the interleaved sts (comment out lines 43-46 and replace line 56 with 57)
    ForwardEuler(fields_ptr->d_all_fields, fields_ptr->d_all_buffer_r);
    

    // if we want to do supertimestepping now we need to transform fields to real explicitely
    // if (param_ptr->supertimestepping and (current_step%2)==0) {
    if (param_ptr->supertimestepping) {
        // this functions copies the complex fields from d_all_fields into d_all_buffer_r and performs
        // an in-place r2c FFT to give the real fields. This buffer is reserved for the real fields!
        supervisor_ptr->Complex2RealFields(fields_ptr->d_all_fields, fields_ptr->d_all_buffer_r, vars.NUM_FIELDS);

        rkl_ptr->compute_cycle(fields_ptr->d_all_fields, fields_ptr->d_all_buffer_r);

    }

    cudaEventRecord(supervisor_ptr->stop_2);
    cudaEventSynchronize(supervisor_ptr->stop_2);
    supervisor_ptr->updateMainLooptime();

    return ;


}



const double gammaRK[3] = {8.0 / 15.0 , 5.0 / 12.0 , 3.0 / 4.0};
const double xiRK[2] = {-17.0 / 60.0 , -5.0 / 12.0};

void TimeStepping::RungeKutta3(data_type* complex_Fields, scalar_type* real_Buffer) {


    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    if (param_ptr->debug > 0) {
        std::printf("Now entering RungeKutta3 function \n");
    }

    // a buffer to temporarily store the dU fields
    data_type* complex_dFields = fields_ptr->d_all_dfields;

    double dt_RK = 0.0;
    int blocksPerGrid = (2 * grid.NTOTAL_COMPLEX * vars.NUM_FIELDS + threadsPerBlock - 1) / threadsPerBlock;



    dt_RK = current_dt; // in theory one can do strang splitting so dt_RK can be 1/2 dt

    if (param_ptr->debug > 1) {
        // std::printf("RK, finished 1st step.\n");
        // std::printf("After compute dfield, RK, 1st step:\n");
        // print_device_values();
        if (current_step == 1 || current_step % 100 == 0 ) std::printf("t: %.5e \t dt: %.5e \n",current_time,dt_RK);
        if (current_step == 1 || current_step % 100 == 0 ) fields_ptr->print_device_values();
    }

    if (param_ptr->debug > 1) {
        std::printf("RK3, doing step n. %d ...\n",stage_step+1);
    }

    compute_dfield(complex_Fields, real_Buffer, complex_dFields);


    // d_all_fields = d_all_fields + gammaRK[0] * dt * d_all_dfields;
    axpyComplex<<<blocksPerGrid, threadsPerBlock>>>( complex_Fields, complex_dFields, complex_Fields, (scalar_type) 1.0, gammaRK[0]*dt_RK,  grid.NTOTAL_COMPLEX * vars.NUM_FIELDS);
    // // d_all_scrtimestep = d_all_fields + xiRK[0] * dt * d_all_dfields;
    axpyComplex<<<blocksPerGrid, threadsPerBlock>>>( complex_Fields, complex_dFields, d_all_scrtimestep, (scalar_type) 1.0, xiRK[0]*dt_RK,  grid.NTOTAL_COMPLEX * vars.NUM_FIELDS);

    // end of stage 1
    stage_step++;

    if (param_ptr->debug > 1) {
        std::printf("RK3, doing step n. %d ...\n",stage_step+1);
    }

    if (param_ptr->shearing) {
        fields_ptr->wavevector.shearWavevector(tremap + gammaRK[0]*dt_RK);
    }

    // std::printf("...Computing dfield\n");
    compute_dfield(complex_Fields, real_Buffer, complex_dFields);

    // for( i = 0 ; i < NTOTAL_COMPLEX ; i++) {
    //         fld.farray[n][i] = fld1.farray[n][i] + gammaRK[1] * dfld.farray[n][i] * dt;
    //         fld1.farray[n][i] = fld.farray[n][i] + xiRK[1] * dfld.farray[n][i] * dt;
    //     }

    // d_all_fields = d_all_scrtimestep + gammaRK[1] * dt * d_all_dfields;
    axpyComplex<<<blocksPerGrid, threadsPerBlock>>>( d_all_scrtimestep, complex_dFields, complex_Fields, (scalar_type) 1.0, gammaRK[1]*dt_RK,  grid.NTOTAL_COMPLEX * vars.NUM_FIELDS);
    // d_all_scrtimestep = d_all_fields + xiRK[1] * dt * d_all_dfields;
    axpyComplex<<<blocksPerGrid, threadsPerBlock>>>( complex_Fields, complex_dFields, d_all_scrtimestep, (scalar_type) 1.0, xiRK[1]*dt_RK,  grid.NTOTAL_COMPLEX * vars.NUM_FIELDS);

    // end of stage 2
    stage_step++;

    if (param_ptr->debug > 1) {
        std::printf("RK3, doing step n. %d ...\n",stage_step+1);
    }

    if (param_ptr->shearing) {
        fields_ptr->wavevector.shearWavevector(tremap + (gammaRK[0] + xiRK[0] + gammaRK[1])*dt_RK);
    }

    // std::printf("...Computing dfield\n");
    compute_dfield(complex_Fields, real_Buffer, complex_dFields);

    // for( i = 0 ; i < NTOTAL_COMPLEX ; i++) {
    //     fld.farray[n][i] = fld1.farray[n][i] + gammaRK[2] * dfld.farray[n][i] * dt;
    // }
    // d_all_fields = d_all_scrtimestep + gammaRK[2] * dt * d_all_dfields;
    axpyComplex<<<blocksPerGrid, threadsPerBlock>>>( d_all_scrtimestep, complex_dFields, complex_Fields, (scalar_type) 1.0, gammaRK[2]*dt_RK,  grid.NTOTAL_COMPLEX * vars.NUM_FIELDS);

    current_time += current_dt;

    if (param_ptr->shearing) {

        tremap += current_dt;
        // is a remap necessary?
        if (tremap > param_ptr->ly / (2.0 * param_ptr->shear * param_ptr->lx)) {
            // if yes recompute tremap
            ShiftTime();
            // and remap fields
            RemapAllFields(complex_Fields);
        }

        fields_ptr->wavevector.shearWavevector(tremap);
    }

    // end of stage 3
    stage_step++;
    if (param_ptr->debug > 1) {
        std::printf("End of RK3 integrator, t: %.5e \t dt: %.5e \n",current_time,current_dt);
    }


}

void TimeStepping::ForwardEuler(data_type* complex_Fields, scalar_type* real_Buffer) {


    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    if (param_ptr->debug > 0) {
        std::printf("Now entering ForwardEuler function \n");
    }

    // a buffer to temporarily store the dU fields
    data_type* complex_dFields = fields_ptr->d_all_dfields;

    double dt_EU = 0.0;
    int blocksPerGrid = (2 * grid.NTOTAL_COMPLEX * vars.NUM_FIELDS + threadsPerBlock - 1) / threadsPerBlock;



    dt_EU = current_dt; // in theory one can do strang splitting so dt_RK can be 1/2 dt

    if (param_ptr->debug > 1) {
        // std::printf("RK, finished 1st step.\n");
        // std::printf("After compute dfield, RK, 1st step:\n");
        // print_device_values();
        if (current_step == 1 || current_step % 100 == 0 ) std::printf("t: %.5e \t dt: %.5e \n",current_time,dt_EU);
        if (current_step == 1 || current_step % 100 == 0 ) fields_ptr->print_device_values();
    }

    compute_dfield(complex_Fields, real_Buffer, complex_dFields);


    // d_all_fields = d_all_fields + gammaRK[0] * dt * d_all_dfields;
    axpyComplex<<<blocksPerGrid, threadsPerBlock>>>( complex_Fields, complex_dFields, complex_Fields, (scalar_type) 1.0, dt_EU,  grid.NTOTAL_COMPLEX * vars.NUM_FIELDS);
    
    
    current_time += current_dt;

    if (param_ptr->shearing) {

        tremap += current_dt;
        // is a remap necessary?
        if (tremap > param_ptr->ly / (2.0 * param_ptr->shear * param_ptr->lx)) {
            // if yes recompute tremap
            ShiftTime();
            // and remap fields
            RemapAllFields(complex_Fields);
        }

        fields_ptr->wavevector.shearWavevector(tremap);
    }

    
    if (param_ptr->debug > 1) {
        std::printf("End of ForwardEuler integrator, t: %.5e \t dt: %.5e \n",current_time,current_dt);
    }


}

