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

void startup();

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

    double t = 0.0;
    // double t_end = 0.0;
    double t_lastsnap = 0.0;
    // int step = 0;
    int num_save = 0;



    // test_forward_inverse_transform();
    // test_do_multiplications();
    // test_axpy();

    std::printf("-----------Initializing fields\n");
    init_plan(fft_size);
    std::printf("Initialized fft\n");
    init_cublas();

    Parameters *param;
    param = new Parameters();
    param->read_Parameters(input_dir);

    if (program.is_used("--output-dir")){
        std::string output_dir = program.get<std::string>("--output-dir");
        std::cout << "output directory will be overriden: " << output_dir << std::endl;
        param->output_dir = output_dir;
    }
    // param->read_Parameters();
    std::printf("Finished reading in params\n");
    // init fields
    Fields fields(NUM_FIELDS, param);
    // fields.init_Fields(param);
    // fields.param.read_Parameters();

    // t_end = param->t_final;
    // Fields fields(3);
    std::printf("lx = %f \t ly = %f \t lz = %f\n",param->lx, param->ly, param->lz);
    std::printf("kxmax = %.2e  kymax = %.2e  kzmax = %.2e \n",fields.wavevector.kxmax,fields.wavevector.kymax, fields.wavevector.kzmax);

    std::printf("nu_th = %.2e \n",param->nu_th);
    std::printf("nu = %.2e \n",param->nu);
    std::printf("N2 = %.2e \n",param->N2);
    std::printf("t_final = %.2e \n",param->t_final);
    // std::printf("t_end = %.2e \n",t_end);
    std::printf("Saving every  dt = %.2e \n",param->toutput_flow);
    // std::printf("Printing host values\n");
#ifdef DEBUG
    fields.wavevector.print_values();
    fields.print_host_values();
#endif

    try {
    fields.write_data_file(num_save);
    }
    catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    // std::cerr << program;
    std::exit(1);
    }
    // fields.write_data_file(num_save, param);
    // wavevector is a member of Fields
    // fields.wavevector.print_values();

    std::printf("Allocating to gpu\n");
    fields.allocate_and_move_to_gpu();

    // fields.print_device_values();


    while (fields.current_time < param->t_final) {
        // std::printf("step n. %d \n",fields.current_step);
    // while (fields.current_step < 1) {

        // dt = fields.advance_timestep(t, t_end, &step); // this function computes dt and advances the time (field(n+1) = field(n) + dfield*dt)
        fields.RungeKutta3();
        // fields.print_device_values();
        // fields.current_time = t;
        if( (fields.current_time-t_lastsnap)>=param->toutput_flow) {
            fields.ComputeDivergence();
            fields.CleanFieldDivergence();
            fields.ComputeDivergence();
            fields.copy_back_to_host();
            fields.write_data_file(num_save+1);
            std::printf("Saving at step n. %d \n",fields.current_step);
            std::printf("Saving data file at t= %.6e \n",fields.current_time);
            t_lastsnap = t_lastsnap + param->toutput_flow;
            num_save++;
        }

    }

    // fields.do_operations();
    // std::printf("Finished operations\n");
    // fields.do_multiplications();
    // std::printf("Finished multiplications\n");
    // fields.print_device_values();

    std::printf("Starting copy back to host\n");
    fields.copy_back_to_host();
    std::printf("Saving data file...\n");
    fields.write_data_file(num_save+1);

    fields.clean_gpu();
    std::printf("Finished fields gpu cleanup\n");

#ifdef DEBUG
    // fields.wavevector.print_values();
    fields.print_host_values();
#endif

    std::printf("Finishing cufft\n");
    finish_cufft();

    std::printf("Finishing cublas\n");
    finish_cublas();

    delete param;

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
