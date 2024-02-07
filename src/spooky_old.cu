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
#include "cuda_kernels.hpp"



// Parameters param;
// int threadsPerBlock = 512;

int main(int argc, char *argv[]) {

    double t = 0.0;
    double t_end = 0.0;
    double t_lastsnap = 0.0;
    // int step = 0;
    int num_save = 0;



    test_forward_inverse_transform();
    // test_do_multiplications();
    // test_axpy();

    std::printf("-----------Initializing fields\n");
    init_plan(fft_size);
    std::printf("Initialized fft\n");
    init_cublas();

    Parameters *param;
    param = new Parameters();
    param->read_Parameters();

    // init fields
    Fields fields(NUM_FIELDS, param);
    // fields.init_Fields(param);
    // fields.param->read_Parameters();

    t_end = param->t_final;
    // Fields fields(3);
    std::printf("lx = %f \t ly = %f \t lz = %f\n",param->lx, param->ly, param->lz);
    std::printf("kxmax = %.2e  kymax = %.2e  kzmax = %.2e \n",fields.wavevector.kxmax,fields.wavevector.kymax, fields.wavevector.kzmax);
    fields.wavevector.print_values();
    std::printf("nu_th = %.2e \n",param->nu_th);
    std::printf("nu = %.2e \n",param->nu);
    std::printf("N2 = %.2e \n",param->N2);
    std::printf("t_final = %.2e \n",param->t_final);
    // std::printf("t_end = %.2e \n",t_end);
    std::printf("Saving every  dt = %.2e \n",param->toutput_flow);
    // std::printf("Printing host values\n");
    fields.print_host_values();
    fields.write_data_file(num_save, param);
    // wavevector is a member of Fields
    // fields.wavevector.print_values();

    std::printf("Allocating to gpu\n");
    fields.allocate_and_move_to_gpu();

    // fields.print_device_values();


    while (t < t_end) {
        // std::printf("step n. %d \n",fields.current_step);
    // while (fields.current_step < 1) {

        // dt = fields.advance_timestep(t, t_end, &step); // this function computes dt and advances the time (field(n+1) = field(n) + dfield*dt)
        fields.RungeKutta3( t,  t_end, param);
        // fields.print_device_values();
        t = t + fields.current_dt;
        fields.current_time = t;
        if( (t-t_lastsnap)>=param->toutput_flow) {
            fields.copy_back_to_host();
            fields.write_data_file(num_save+1, param);
            std::printf("Saving at step n. %d \n",fields.current_step);
            std::printf("Saving data file at t= %.6e \n",t);
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
    fields.write_data_file(num_save+1, param);

    fields.clean_gpu();
    std::printf("Finished fields gpu cleanup\n");

    fields.print_host_values();

    std::printf("Finishing cufft\n");
    finish_cufft();

    std::printf("Finishing cublas\n");
    finish_cublas();

    return EXIT_SUCCESS;
};
