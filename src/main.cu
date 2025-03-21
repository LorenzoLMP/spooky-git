#include "common.hpp"
#include <cuda_runtime.h>
#include "cufft_utils.h"
#include "cufft_routines.hpp"
#include "cublas_routines.hpp"
#include <complex.h>
#include "fields.hpp"
#include <argparse/argparse.hpp>
#include "parameters.hpp"
#include "inputoutput.hpp"
#include "timestepping.hpp"
#include "physics.hpp"
#include "supervisor.hpp"

// #define SET 0
// #define ADD 1

void startup();

Variables vars;
Grid grid;
int threadsPerBlock{512};

int main(int argc, char *argv[]) {

    int restart_num = -1;
    // int stats_frequency = -1;

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

    program.add_argument("-t", "--time")
    .help("override the maximum wallclock elapsed time (format '-t HH MM SS' where HH, MM, SS are 3 integers for hours, seconds and minutes): ")
    .nargs(3)
    .scan<'i', int>()
    .default_value(std::vector<int>{0, 0, 10});

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



    startup();

    //----------------------------------------------------------------------------------------
    //! Initialize objects

    std::printf("-----------Initializing objects...\n");

    Supervisor spooky(input_dir);
    // Supervisor spooky(stats_frequency);
    //
    // Parameters param(input_dir);
    // Fields fields(param, vars.NUM_FIELDS);
    // Physics phys(spooky);
    // TimeStepping timestep(vars.NUM_FIELDS, param, spooky);
    // InputOutput inout(spooky);

    std::printf("grid size: %d %d %d  \n",int(grid.FFT_SIZE[0]),int(grid.FFT_SIZE[1]),int(grid.FFT_SIZE[2]));

    std::printf("-----------Initializing cufft, cublas...\n");

    init_plan(grid.FFT_SIZE);
    init_cublas();

    std::printf("Finished reading in params and initializing objects.\n");


    //----------------------------------------------------------------------------------------
    //! Parse runtime flags and override default params

    if (program.is_used("--output-dir")){
        std::string output_dir = program.get<std::string>("--output-dir");
        std::cout << "output directory will be overriden: " << output_dir << std::endl;
        spooky.param_ptr->output_dir = output_dir;
    }
    if (program.is_used("--restart")){
        // std::cout << "restarting from file: "  << std::endl;
        restart_num = program.get<int>("--restart");
        std::cout << "restarting from file: " << restart_num << std::endl;
        spooky.param_ptr->restart = 1;
    }
    if (program.is_used("--stats")){
        spooky.stats_frequency = program.get<int>("--stats");
        std::cout << "printing stats every " << spooky.stats_frequency << " steps " << std::endl;
    }
    if (program.is_used("--time")){
        std::vector<int> max_walltime_elapsed = program.get<std::vector<int>>("--time");
        double max_hours = double(max_walltime_elapsed[0]) + double(max_walltime_elapsed[1])/60 + double(max_walltime_elapsed[2])/3600;
        std::cout << "overriding wallclock max elapsed time: " << max_walltime_elapsed[0] << " hours " << max_walltime_elapsed[1] << " minutes " << max_walltime_elapsed[2] << " seconds " << std::endl;
        std::cout << "... in hours: " << max_hours << std::endl;
        spooky.param_ptr->max_walltime_elapsed = 0.95*max_hours;
    }

    spooky.Restart(restart_num);

    spooky.displayConfiguration();


    if (spooky.param_ptr->debug > 1) {
        spooky.fields_ptr->wavevector.print_values();
        spooky.fields_ptr->print_host_values();
    }

    std::printf("Allocating to gpu...\n");
    spooky.fields_ptr->allocate_and_move_to_gpu();

    spooky.fields_ptr->CheckSymmetries();

    spooky.initialDataDump();

    spooky.executeMainLoop();

    spooky.print_final_stats();

    // std::printf("Starting copy back to host\n");
    // spooky.fields_ptr->copy_back_to_host();
    

    spooky.fields_ptr->clean_gpu();
    std::printf("Finished fields gpu cleanup\n");

    // if (spooky.param_ptr->debug > 1) {
    //     spooky.fields_ptr->print_host_values();
    // }

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



