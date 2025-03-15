#include "common.hpp"
#include "supervisor.hpp"
#include "parameters.hpp"
#include "fields.hpp"
#include "inputoutput.hpp"
#include "timestepping.hpp"
#include "physics.hpp"

#include "cublas_routines.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "cufft_routines.hpp"

#include <cuda_runtime.h>
// #include <cufftXt.h>
#include "cufft_utils.h"


Supervisor::Supervisor(std::string input_dir) :
        total_timer() ,
        timevar_timer(),
        datadump_timer() {

    param_ptr = std::shared_ptr<Parameters> (new Parameters(*this, input_dir));

    /*****
     *
     * Populate Variables/Grid struct
     *
     * ****/
    param_ptr->popVariablesGrid();

    if (not param_ptr->checkParameters()){
        std::cout << "Error: your choice of physics modules is not consistent. Aborting now." << std::endl;
        exit(0);
    }


    fields_ptr = std::shared_ptr<Fields> (new Fields(*this, *param_ptr));
    phys_ptr = std::shared_ptr<Physics> (new Physics(*this));
    timestep_ptr = std::shared_ptr<TimeStepping> (new TimeStepping(*this, *param_ptr));
    inout_ptr = std::shared_ptr<InputOutput> (new InputOutput(*this));


    stats_frequency = -1;
    time_delta = 0.0;
    NumFFTs = 0; // in mainloop
    // NumFFTs[1] = 0; // in IO
    TimeSpentInFFTs = 0.0;
    TimeSpentInMainLoop = 0.0;
    TimeSpentInMainLoopPartial = 0.0;

    TimeIOTimevar = 0.0;
    TimeIODatadump = 0.0;

    AllocCpuMem = 0;
    AllocGpuMem = 0;

    ElapsedWallClockTime = 0.0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventCreate(&start_2);
    cudaEventCreate(&stop_2);
}

void Supervisor::updateFFTtime(){
    // in ms
    cudaEventElapsedTime(&time_delta, start, stop);
    // in s
    TimeSpentInFFTs += 1e-3*time_delta;
}

void Supervisor::updateMainLooptime(){
    // in ms
    cudaEventElapsedTime(&time_delta_2, start_2, stop_2);
    // in s
    TimeSpentInMainLoop += 1e-3*time_delta_2;
    TimeSpentInMainLoopPartial += 1e-3*time_delta_2;
}

void Supervisor::print_partial_stats(){

    std::printf("---- The avg number of cell updates / sec is %.4e [cell_updates/s]  \n",grid.NTOTAL*stats_frequency/TimeSpentInMainLoopPartial);

    TimeSpentInMainLoopPartial = 0.0;
}

void Supervisor::print_final_stats(){

    int tot_steps = timestep_ptr->current_step;
    std::printf("@@@@@ ------------------------------------------------------------------ @@@@@ \n");
    std::printf("@@ ------------------------------------------------------------------------ @@ \n");
    std::printf("@@\t \t \t FINAL STATISTICS REPORT \n");
    std::printf("@@\tThe total execution time was: \t\t\t %.2e [s]  \n", total_timer.elapsed());
    std::printf("@@\tThe time spent in the mainloop was: \t\t %.2e [s]  \n",TimeSpentInMainLoop);
    std::printf("@@\tThe time spent in FFTs in the mainloop was: \t %.2e [s]  \n", TimeSpentInFFTs);
    std::printf("@@\tThe time spent in IO (timevar+datadumps): \t %.2e [s]  \n", TimeIOTimevar+TimeIODatadump);
    std::printf("@@\t\t- time spent in timevar: \t\t %.2e [s]  \n", TimeIOTimevar);
    std::printf("@@\t\t- time spent in datadumps: \t\t %.2e [s]  \n",TimeIODatadump);

    std::cout << "@@" << std::endl;
    std::printf("@@\tThe mainloop took %d FFTs and %d steps to complete \n", NumFFTs, tot_steps);
    std::printf("@@\t\t- FFTs per loop: %d \n", NumFFTs/tot_steps);
    std::cout << "@@" << std::endl;
    std::printf("@@\tThe average performance is: \n@@\t\t\t %.4e [cell_updates/s]  \n",grid.NTOTAL*tot_steps/TimeSpentInMainLoop);

    std::printf("@@ ------------------------------------------------------------------------ @@ \n");
    std::printf("@@@@@ ------------------------------------------------------------------ @@@@@ \n");
}

void Supervisor::displayConfiguration(){

    std::printf("You are running SPOOKY with the following configuration: \n");
    std::printf("Number of fields = %d \n",vars.NUM_FIELDS);
    std::printf("Grid size (real space) (NX, NY, NZ) = (%d \t %d \t %d)\n",grid.NX, grid.NY, grid.NZ);
    std::printf("Size of the domain: lx = %f \t ly = %f \t lz = %f\n",param_ptr->lx, param_ptr->ly, param_ptr->lz);
    std::printf("kxmax = %.4e  kymax = %.4e  kzmax = %.4e \n",fields_ptr->wavevector.kxmax,fields_ptr->wavevector.kymax, fields_ptr->wavevector.kzmax);

    if (param_ptr->incompressible) {
        std::printf("Re = %.4e \n",param_ptr->reynolds);
    }
    if (param_ptr->boussinesq or param_ptr->heat_equation) {
        std::printf("Re_th = %.4e \n",param_ptr->reynolds_th);
    }
    if (param_ptr->mhd) {
        std::printf("Re_m = %.4e \n",param_ptr->reynolds_m);
    }
    if (param_ptr->anisotropic_diffusion) {
        std::printf("Re_aniso = %.4e \n",param_ptr->reynolds_ani);
        std::printf("Omega_T2 = %.4e \n",param_ptr->OmegaT2);
    }

    if (param_ptr->stratification) {
        std::printf("Background stratification in direction: %d \n",param_ptr->strat_direction);
        std::printf("N2 = %.4e \n",param_ptr->N2);

    }
    if (param_ptr->shearing) {
        std::printf("S = %.4e \n",param_ptr->shear);
    }
    if (param_ptr->rotating) {
        std::printf("Omega = %.4e \n",param_ptr->omega);
    }

    if (param_ptr->supertimestepping) {
        // std::printf("Algorithm for supertimestepping: %s \n",param_ptr->sts_algorithm);
        std::cout << "Algorithm for supertimestepping: " << param_ptr->sts_algorithm << std::endl;
    }

    std::printf("t_initial = %.4e \n",param_ptr->t_initial);
    std::printf("t_current = %.4e \n",timestep_ptr->current_time);
    std::printf("t_final = %.4e \n",param_ptr->t_final);

    std::printf("Enforcing symmetries every %d steps \n",param_ptr->symmetries_step);
    std::printf("Saving snapshot every  dt = %.2e \n",param_ptr->toutput_flow);
    std::printf("Saving timevar every  dt = %.2e \n",param_ptr->toutput_time);
    std::printf("Displaying stats every num steps = %d \n",stats_frequency);
    std::printf("Maximum wallclock elapsed time (in hours): %.4e \n", param_ptr->max_walltime_elapsed);

}

void Supervisor::executeMainLoop(){

    if (param_ptr->shearing) {
        timestep_ptr->ShiftTime();
        fields_ptr->wavevector.shearWavevector(timestep_ptr->tremap);
        std::printf("t_remap = %.4e \n",timestep_ptr->tremap);
    }

    // TimeSpentInMainLoop is in seconds while
    // max_walltime_elapsed is in hours
    while (timestep_ptr->current_time < param_ptr->t_final and TimeSpentInMainLoop < param_ptr->max_walltime_elapsed*3600) {

        // advance the equations (field(n+1) = field(n) + dfield*dt)
        // timestep_ptr->RungeKutta3();
        timestep_ptr->HydroMHDAdvance(fields_ptr);
        // check if we need to output data
        inout_ptr->CheckOutput();
        // check if we need to enforce symmetries
        fields_ptr->CheckSymmetries();

        if (param_ptr->debug == 2){
            std::printf("step: %d \t dt: %.2e \n", timestep_ptr->current_step,timestep_ptr->current_dt);
        }

        if (stats_frequency > 0){
            if ( timestep_ptr->current_step % stats_frequency == 0)
            print_partial_stats();
        }

        if (TimeSpentInMainLoop >= param_ptr->max_walltime_elapsed*3600){
            std::printf("The maximum wallclock elapsed time was reached. Stopping now. You can later resume from the last snapshot available. \n");
        }


    }
}

void Supervisor::initialDataDump(){

    if (param_ptr->restart == 0){

        std::printf("Initial data dump...\n");
        try {
        inout_ptr->CheckOutput();
        }
        catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::exit(1);
        }
    }
}


void Supervisor::Restart(int restart_num){
    if (param_ptr->restart == 1){
        inout_ptr->ReadDataFile(restart_num);
    }
}

Supervisor::~Supervisor(){

}

void Supervisor::Complex2RealFields(data_type* ComplexField_in, int num_fields){

    // version with in-place transform
    // compute FFTs from complex to real fields
    for (int n = 0; n < num_fields; n++){
        c2r_fft(ComplexField_in + n * grid.NTOTAL_COMPLEX,  ((scalar_type*) ComplexField_in) + n * 2*grid.NTOTAL_COMPLEX, this);
    }

}

void Supervisor::Complex2RealFields(data_type* ComplexField_in, scalar_type* RealField_out, int num_fields){

    // assign fields to [num_fields] tmparray (memory block starts at d_all_tmparray)
    int blocksPerGrid = ( num_fields * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
    ComplexVecAssign<<<blocksPerGrid, threadsPerBlock>>>(ComplexField_in, (data_type*) RealField_out, num_fields * grid.NTOTAL_COMPLEX);

    // compute FFTs from complex to real fields
    for (int n = 0; n < num_fields; n++){
        c2r_fft((data_type*) RealField_out + n * grid.NTOTAL_COMPLEX,  RealField_out + n * 2*grid.NTOTAL_COMPLEX, this);
    }

}
