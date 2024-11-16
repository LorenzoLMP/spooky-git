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

    Supervisor spooky(input_dir, stats_frequency);
    // Supervisor spooky(stats_frequency);
    //
    // Parameters param(input_dir);
    // Fields fields(param, NUM_FIELDS);
    // Physics phys(spooky);
    // TimeStepping timestep(NUM_FIELDS, param, spooky);
    // InputOutput inout(spooky);


    std::printf("Finished reading in params and initializing objects.\n");


    //----------------------------------------------------------------------------------------
    //! Parse runtime flags and override default params

    if (program.is_used("--output-dir")){
        std::string output_dir = program.get<std::string>("--output-dir");
        std::cout << "output directory will be overriden: " << output_dir << std::endl;
        spooky.param->output_dir = output_dir;
    }
    if (program.is_used("--restart")){
        // std::cout << "restarting from file: "  << std::endl;
        restart_num = program.get<int>("--restart");
        std::cout << "restarting from file: " << restart_num << std::endl;
        spooky.param->restart = 1;
    }

    spooky.displayConfiguration();

    spooky.Restart(restart_num);

#ifdef DDEBUG
    spooky.fields->wavevector.print_values();
    spooky.fields->print_host_values();
#endif

    std::printf("Allocating to gpu...\n");
    spooky.fields->allocate_and_move_to_gpu();

    spooky.fields->CheckSymmetries();

    spooky.initialDataDump();

    // wavevector is a member of Fields
    // spooky.fields->wavevector.print_values();

    spooky.executeMainLoop();


    spooky.print_final_stats();

    // std::printf("Starting copy back to host\n");
    spooky.fields->copy_back_to_host();
    

    spooky.fields->clean_gpu();
    std::printf("Finished fields gpu cleanup\n");

#ifdef DDEBUG
    // fields->wavevector.print_values();
    spooky.fields->print_host_values();
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



