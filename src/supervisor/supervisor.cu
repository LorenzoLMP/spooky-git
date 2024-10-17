#include "define_types.hpp"
#include "supervisor.hpp"
// #include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "cublas_routines.hpp"
#include "cuda_kernels.hpp"
#include "parameters.hpp"
// #include "inputoutput.hpp"
#include "fields.hpp"
#include <cuda_runtime.h>
// #include <cufftXt.h>
// #include "spooky.hpp"
#include "cufft_utils.h"
// #include "define_types.hpp"

Supervisor::Supervisor(int stats_frequency) : stats_frequency(stats_frequency) , total_timer() , timevar_timer(), datadump_timer() {

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

void Supervisor::print_final_stats(int tot_steps){

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

Supervisor::~Supervisor(){

}
