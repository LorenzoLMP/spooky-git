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
Parser parser;

int threadsPerBlock{512};

int main(int argc, char *argv[]) {

    argparse::ArgumentParser program("spooky");

    program.add_argument("--input-dir")
    .help("input directory for cfg file")
    .default_value(std::string("./"));

    program.add_argument("--output-dir")
    // .required()
    .help("output directory for data files");

    program.add_argument("-r", "--restart")
    .help("restart from data file")
    .scan<'i', int>()
    .default_value(int(-1));

    program.add_argument("--stats")
    .help("whether to print stats: -1 (none), n > 0 (every n steps)")
    .scan<'i', int>()
    .default_value(int(-1));

    program.add_argument("-t", "--time")
    .help("override the maximum wallclock elapsed time (format '-t HH MM SS' where HH, MM, SS are 3 integers for hours, seconds and minutes): ")
    .nargs(3)
    .scan<'i', int>()
    .default_value(std::vector<int>{0, 0, 10});

    program.add_argument("-n", "--ngrid")
    .help("override the number of grid points (format '-n nx ny nz' where nx, ny, nz are 3 integers for the number of cells in the x, y, and z direction): ")
    .nargs(3)
    .scan<'i', int>()
    .default_value(std::vector<int>{32, 32, 32});

    try {
    program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
    }

    


    startup();

    
    // ----------------------------------------------------------------------------------------
    //! Parse runtime flags and override default params

    parser.input_dir = program.get<std::string>("--input-dir");
    std::cout << "Input directory: " << parser.input_dir << std::endl;

    if (program.is_used("--output-dir")){
        // this will override the output dir
        // in the spooky.cfg file 
        parser.output_dir = program.get<std::string>("--output-dir");
        parser.output_dir_override = true;
        std::cout << "output directory will be overriden: " << parser.output_dir << std::endl;
    }

    if (program.is_used("--restart")){
        parser.restart_num = program.get<int>("--restart");
    // if (parser.restart_num >= -1) {
        std::cout << "restarting from file: " << parser.restart_num << std::endl;
    }
    else {
        // means not restart
        parser.restart_num = -2;
    }
    
    parser.stats_frequency = program.get<int>("--stats");
    if (parser.stats_frequency > -1){
        std::cout << "printing stats every " << parser.stats_frequency << " steps " << std::endl;
    }
    
    if (program.is_used("--time")){
        // this will override the max walltime elapsed 
        // in the spooky.cfg file 
        std::vector<int> max_walltime_elapsed = program.get<std::vector<int>>("--time");
        parser.max_hours = double(max_walltime_elapsed[0]) + double(max_walltime_elapsed[1])/60 + double(max_walltime_elapsed[2])/3600;

        std::cout << "overriding wallclock max elapsed time: " << max_walltime_elapsed[0] << " hours " << max_walltime_elapsed[1] << " minutes " << max_walltime_elapsed[2] << " seconds " << std::endl;
        std::cout << "... in hours: " << parser.max_hours << std::endl;
        std::cout << "(we take 95pc of that and terminate earlier)" << std::endl;
        parser.max_hours *= 0.95;
    }

    if (program.is_used("--ngrid")){
        // this will override the max walltime elapsed 
        // in the spooky.cfg file 
        std::vector<int> ngrid = program.get<std::vector<int>>("--ngrid");
        parser.nx = ngrid[0];
        parser.ny = ngrid[1];
        parser.nz = ngrid[2];

        std::printf("overriding ngrid: (nx, ny, nz) = (%d, %d, %d) \n", parser.nx, parser.ny, parser.nz);
    }



    //----------------------------------------------------------------------------------------
    //! Initialize objects

    std::printf("-----------Initializing objects...\n");

    Supervisor spooky;

    std::printf("grid size: %d %d %d  \n",int(grid.FFT_SIZE[0]),int(grid.FFT_SIZE[1]),int(grid.FFT_SIZE[2]));

    std::printf("-----------Initializing cufft, cublas...\n");

    init_plan(grid.FFT_SIZE);
    init_cublas();

    std::printf("Finished reading in params and initializing objects.\n");

    if (parser.restart_num >= -1) {
        spooky.Restart();
    }

    spooky.displayConfiguration();


    if (spooky.param_ptr->debug > 1) {
        spooky.fields_ptr->wavevector.print_values();
        spooky.fields_ptr->print_host_values();
    }

    std::printf("Allocating to gpu...\n");
    spooky.fields_ptr->allocate_and_move_to_gpu();

    spooky.fields_ptr->CheckSymmetries();

    spooky.initialDataDump();

    //----------------------------------------------------------------------------------------
    //! Main Loop

    spooky.executeMainLoop();

    spooky.print_final_stats();

    //----------------------------------------------------------------------------------------
    //! Finalize

    spooky.fields_ptr->clean_gpu();
    std::printf("Finished fields gpu cleanup\n");

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



