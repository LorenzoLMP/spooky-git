#include "define_types.hpp"
#include "supervisor.hpp"
#include "parameters.hpp"
#include "fields.hpp"
#include "inputoutput.hpp"
#include "timestepping.hpp"
#include "physics.hpp"


// #include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "cublas_routines.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "cufft_routines.hpp"

#include <cuda_runtime.h>
// #include <cufftXt.h>
// #include "spooky.hpp"
#include "cufft_utils.h"
// #include "define_types.hpp"

Supervisor::Supervisor(std::string input_dir) :
        total_timer() ,
        timevar_timer(),
        datadump_timer() {

    param_ptr = std::shared_ptr<Parameters> (new Parameters(*this, input_dir));
    fields_ptr = std::shared_ptr<Fields> (new Fields(*this, *param_ptr));
    phys_ptr = std::shared_ptr<Physics> (new Physics(*this));
    timestep_ptr = std::shared_ptr<TimeStepping> (new TimeStepping(*this, *param_ptr));
    inout_ptr = std::shared_ptr<InputOutput> (new InputOutput(*this));

    // param(this, input_dir),
    // fields(this, param),
    // phys(this),
    // timestep(this, param),
    // inout(this),

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

    std::printf("---- The avg number of cell updates / sec is %.4e [cell_updates/s]  \n",ntotal*stats_frequency/TimeSpentInMainLoopPartial);

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
    std::printf("@@\tThe average performance is: \n@@\t\t\t %.4e [cell_updates/s]  \n",ntotal*tot_steps/TimeSpentInMainLoop);

    std::printf("@@ ------------------------------------------------------------------------ @@ \n");
    std::printf("@@@@@ ------------------------------------------------------------------ @@@@@ \n");
}

void Supervisor::displayConfiguration(){

    std::printf("lx = %f \t ly = %f \t lz = %f\n",param_ptr->lx, param_ptr->ly, param_ptr->lz);
    std::printf("kxmax = %.2e  kymax = %.2e  kzmax = %.2e \n",fields_ptr->wavevector.kxmax,fields_ptr->wavevector.kymax, fields_ptr->wavevector.kzmax);
    std::printf("numfields = %d",fields_ptr->num_fields);
#ifdef BOUSSINESQ
    std::printf("nu_th = %.2e \n",param_ptr->nu_th);
#endif
    std::printf("nu = %.2e \n",param_ptr->nu);
#ifdef STRATIFICATION
    std::printf("N2 = %.2e \n",param_ptr->N2);
#endif
    std::printf("t_final = %.2e \n",param_ptr->t_final);
    std::printf("Enforcing symmetries every %d steps \n",param_ptr->symmetries_step);
    std::printf("Saving snapshot every  dt = %.2e \n",param_ptr->toutput_flow);
    std::printf("Saving timevar every  dt = %.2e \n",param_ptr->toutput_time);
    std::printf("Displaying stats every num steps = %d \n",stats_frequency);
}

void Supervisor::executeMainLoop(){


    while (timestep_ptr->current_time < param_ptr->t_final) {

        // advance the equations (field(n+1) = field(n) + dfield*dt)
        // timestep_ptr->RungeKutta3();
        timestep_ptr->HydroMHDAdvance(fields_ptr);
        // check if we need to output data
        inout_ptr->CheckOutput();
        // check if we need to enforce symmetries
        fields_ptr->CheckSymmetries();
#ifdef DDEBUG
        std::printf("step: %d \t dt: %.2e \n", timestep_ptr->current_step,timestep_ptr->current_dt);
#endif

        if (stats_frequency > 0){
            if ( timestep_ptr->current_step % stats_frequency == 0)
            print_partial_stats();
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

void Supervisor::Complex2RealFields(data_type* ComplexField_in, scalar_type* RealField_out, int num_fields){

    // assign fields to [num_fields] tmparray (memory block starts at d_all_tmparray)
    int blocksPerGrid = ( num_fields * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    ComplexVecAssign<<<blocksPerGrid, threadsPerBlock>>>(ComplexField_in, (data_type*) RealField_out, num_fields * ntotal_complex);

    // compute FFTs from complex to real fields
    for (int n = 0; n < num_fields; n++){
        c2r_fft((data_type*) RealField_out + n * ntotal_complex,  RealField_out + n * 2*ntotal_complex, this);
    }

}
