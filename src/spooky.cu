#include "spooky.hpp"
#include <cuda_runtime.h>
// #include <cufftXt.h>
#include "cufft_utils.h"
#include "cufft_routines.hpp"
#include "cublas_routines.hpp"
#include "tests.hpp"
#include <complex.h>
#include "fields.hpp"
// #include "wavevector.hpp"
#include "common.hpp"
// #include "parameters.hpp"
// #include "cuda_kernels.hpp"
// #include "cuda_kernels_generic.hpp"
#include <argparse/argparse.hpp>
#include "parameters.hpp"
#include "inputoutput.hpp"
#include "timestepping.hpp"
#include "physics.hpp"
#include "supervisor.hpp"

void startup();
void displayConfiguration(Fields &fields, Parameters &param);

// Parameters param;
// int threadsPerBlock = 512;

int main(int argc, char *argv[]) {

    int restart_num = -1;
    int stats_frequency = -1;

    argparse::ArgumentParser program("spooky");

    program.add_argument("--input-dir")
    .help("input directory for cfg file")
    .default_value(std::string("./"));

    program.add_argument("--output-dir")
    .help("output directory for data files");

    program.add_argument("-r", "--restart")
    .help("restart from data file")
    .scan<'i', int>();
    // .default_value(int(-1));

    program.add_argument("--stats")
    .help("whether to print stats: -1 (none), n > 0 (every n steps)")
    .scan<'i', int>()
    .default_value(int(-1));


    try {
    program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
    }

    std::string input_dir = program.get<std::string>("--input-dir");
    std::cout << "Input directory: " << input_dir << std::endl;

    if (program.is_used("--stats")){
        stats_frequency = program.get<int>("--stats");
        std::cout << "printing stats every " << stats_frequency << " steps " << std::endl;
    }

    startup();
    
    std::printf("-----------Initializing cufft, cublas...\n");

    init_plan(fft_size);
    init_cublas();

    //----------------------------------------------------------------------------------------
    //! Initialize objects

    std::printf("-----------Initializing objects...\n");

    Supervisor supervisor(stats_frequency);

    Parameters param(input_dir);
    Fields fields(param, NUM_FIELDS);
    Physics phys(supervisor);
    TimeStepping timestep(NUM_FIELDS, param, supervisor);
    InputOutput inout(supervisor);


    std::printf("Finished reading in params and initializing objects.\n");


    //----------------------------------------------------------------------------------------
    //! Parse runtime flags and override default params

    if (program.is_used("--output-dir")){
        std::string output_dir = program.get<std::string>("--output-dir");
        std::cout << "output directory will be overriden: " << output_dir << std::endl;
        param.output_dir = output_dir;
    }
    if (program.is_used("--restart")){
        // std::cout << "restarting from file: "  << std::endl;
        restart_num = program.get<int>("--restart");
        std::cout << "restarting from file: " << restart_num << std::endl;
        param.restart = 1;
    }

    displayConfiguration(fields, param);

    if (param.restart == 1){
        inout.ReadDataFile(fields, param, timestep, restart_num);
    }

#ifdef DDEBUG
    fields.wavevector.print_values();
    fields.print_host_values();
#endif

    std::printf("Allocating to gpu...\n");
    fields.allocate_and_move_to_gpu();

    // fields.print_device_values();

    fields.CheckSymmetries(timestep.current_step, param.symmetries_step);

    if (param.restart == 0){

        std::printf("Initial data dump...\n");
        try {
        inout.CheckOutput(fields, param, timestep);
        }
        catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
        }
    }


    while (timestep.current_time < param.t_final) {

        // advance the equations (field(n+1) = field(n) + dfield*dt)
        timestep.RungeKutta3(fields, param, phys);
        // check if we need to output data
        inout.CheckOutput(fields, param, timestep);
        // check if we need to enforce symmetries
        fields.CheckSymmetries(timestep.current_step, param.symmetries_step);
#ifdef DDEBUG
        std::printf("step: %d \t dt: %.2e \n", timestep.current_step,timestep.current_dt);
#endif

        if (stats_frequency > 0){
            if ( timestep.current_step % stats_frequency == 0)
            supervisor.print_partial_stats();
        }


    }


    supervisor.print_final_stats(timestep.current_step);

    // std::printf("Starting copy back to host\n");
    fields.copy_back_to_host();
    

    fields.clean_gpu();
    std::printf("Finished fields gpu cleanup\n");

#ifdef DDEBUG
    // fields.wavevector.print_values();
    fields.print_host_values();
#endif

    std::printf("Finishing cufft\n");
    finish_cufft();

    std::printf("Finishing cublas\n");
    finish_cublas();

    return EXIT_SUCCESS;
};


void startup(){
	std::cout <<
R"abcd(
                 ____________
               --            --
             /                  \\
            /                    \\
           /     __               \\
          |     /  \       __      ||
          |    |    |     /  \     ||
                \__/      \__/
         |             ^            ||
         |                          ||
         |                          ||
        |                            ||
        |                            ||
        |                            ||
         \__         ______       __//
            \       //     \_____//
             \_____//

)abcd" << std::endl;
}


void displayConfiguration(Fields &fields, Parameters &param){

    std::printf("lx = %f \t ly = %f \t lz = %f\n",param.lx, param.ly, param.lz);
    std::printf("kxmax = %.2e  kymax = %.2e  kzmax = %.2e \n",fields.wavevector.kxmax,fields.wavevector.kymax, fields.wavevector.kzmax);
    std::printf("numfields = %d",fields.num_fields);
#ifdef BOUSSINESQ
    std::printf("nu_th = %.2e \n",param.nu_th);
#endif
    std::printf("nu = %.2e \n",param.nu);
#ifdef STRATIFICATION
    std::printf("N2 = %.2e \n",param.N2);
#endif
    std::printf("t_final = %.2e \n",param.t_final);
    std::printf("Enforcing symmetries every %d steps \n",param.symmetries_step);
    std::printf("Saving snapshot every  dt = %.2e \n",param.toutput_flow);
    std::printf("Saving timevar every  dt = %.2e \n",param.toutput_time);
}
