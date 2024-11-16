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



#include <cuda_runtime.h>
// #include <cufftXt.h>
// #include "spooky.hpp"
#include "cufft_utils.h"
// #include "define_types.hpp"

Supervisor::Supervisor(std::string input_dir, int stats_frequency) :
        stats_frequency(stats_frequency) ,
        total_timer() ,
        timevar_timer(),
        datadump_timer() {

    param = std::shared_ptr<Parameters> (new Parameters(*this, input_dir));
    fields = std::shared_ptr<Fields> (new Fields(*this, param));
    phys = std::shared_ptr<Physics> (new Physics(*this));
    timestep = std::shared_ptr<TimeStepping> (new TimeStepping(*this, param));
    inout = std::shared_ptr<InputOutput> (new InputOutput(*this));

    // param(this, input_dir),
    // fields(this, param),
    // phys(this),
    // timestep(this, param),
    // inout(this),

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

    int tot_steps = timestep->current_step;
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

    std::printf("lx = %f \t ly = %f \t lz = %f\n",param->lx, param->ly, param->lz);
    std::printf("kxmax = %.2e  kymax = %.2e  kzmax = %.2e \n",fields->wavevector.kxmax,fields->wavevector.kymax, fields->wavevector.kzmax);
    std::printf("numfields = %d",fields->num_fields);
#ifdef BOUSSINESQ
    std::printf("nu_th = %.2e \n",param->nu_th);
#endif
    std::printf("nu = %.2e \n",param->nu);
#ifdef STRATIFICATION
    std::printf("N2 = %.2e \n",param->N2);
#endif
    std::printf("t_final = %.2e \n",param->t_final);
    std::printf("Enforcing symmetries every %d steps \n",param->symmetries_step);
    std::printf("Saving snapshot every  dt = %.2e \n",param->toutput_flow);
    std::printf("Saving timevar every  dt = %.2e \n",param->toutput_time);
}

void Supervisor::executeMainLoop(){

    while (timestep->current_time < param->t_final) {

        // advance the equations (field(n+1) = field(n) + dfield*dt)
        timestep->RungeKutta3();
        // check if we need to output data
        inout->CheckOutput();
        // check if we need to enforce symmetries
        fields->CheckSymmetries();
#ifdef DDEBUG
        std::printf("step: %d \t dt: %.2e \n", timestep->current_step,timestep->current_dt);
#endif

        if (stats_frequency > 0){
            if ( timestep->current_step % stats_frequency == 0)
            print_partial_stats();
        }


    }
}

void Supervisor::initialDataDump(){

    if (param->restart == 0){

        std::printf("Initial data dump...\n");
        try {
        inout->CheckOutput();
        }
        catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::exit(1);
        }
    }
}


void Supervisor::Restart(int restart_num){
    if (param->restart == 1){
        inout->ReadDataFile(fields, param, timestep, restart_num);
    }
}

Supervisor::~Supervisor(){

}
