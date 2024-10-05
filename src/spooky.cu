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

void startup();
void displayConfiguration(Fields &fields, Parameters &param);

// Parameters param;
// int threadsPerBlock = 512;

int main(int argc, char *argv[]) {

    argparse::ArgumentParser program("spooky");

    program.add_argument("--input-dir")
    .help("input directory for cfg file")
    .default_value(std::string("./"));

    program.add_argument("--output-dir")
    .help("output directory for data files");


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
    
    std::printf("-----------Initializing fields\n");
    init_plan(fft_size);
    std::printf("Initialized fft\n");
    init_cublas();

    //----------------------------------------------------------------------------------------
    //! Declare classes

    Parameters param(input_dir);
    Fields fields(param, NUM_FIELDS);
    Physics phys;
    TimeStepping timestep(NUM_FIELDS);
    InputOutput inout;


    //----------------------------------------------------------------------------------------
    //! Create instances

    // param = new Parameters(fields, input_dir);
    // param.read_Parameters(input_dir);
    std::printf("Finished reading in params\n");

    if (program.is_used("--output-dir")){
        std::string output_dir = program.get<std::string>("--output-dir");
        std::cout << "output directory will be overriden: " << output_dir << std::endl;
        param.output_dir = output_dir;
    }

    // timestep = new TimeStepping(param, fields, NUM_FIELDS);
    // fields = new Fields(param, timestep, NUM_FIELDS);
    // // fields.init_Fields(NUM_FIELDS, param);
    // inout = new InputOutput(param, timestep, fields);



    displayConfiguration(fields, param);

#ifdef DDEBUG
    fields.wavevector.print_values();
    fields.print_host_values();
#endif

    std::printf("Allocating to gpu...\n");
    fields.allocate_and_move_to_gpu();

    // fields.print_device_values();

    fields.CheckSymmetries(timestep.current_step, param.symmetries_step);

    std::printf("Initial data dump...\n");
    try {
    // fields.CheckOutput();
    inout.write_data_file(fields, param, timestep);
    inout.num_save++;
    inout.write_data_output_header(param);
    inout.write_data_output(fields, param, timestep);
    }
    catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
    }

    // wavevector is a member of Fields
    // fields.wavevector.print_values();

    // std::printf("Current dt: %.2e \n", timestep.current_dt);
    // std::printf("Current time: %.2e \n", timestep.current_time);
    // int n = 0;
    // while (n < 1){
        // timestep.RungeKutta3();
        // std::printf("Current dt: %.2e \n", timestep.current_dt);
        // std::printf("Current time: %.2e \n", timestep.current_time);
        // timestep.current_time = 0.5;
        // std::printf("Current time: %.2f \n", timestep.current_time);
        // n++;
    // }
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
    }

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
