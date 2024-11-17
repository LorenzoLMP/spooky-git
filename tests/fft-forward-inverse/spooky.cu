#include "spooky.hpp"
#include <cuda_runtime.h>
// #include <cufftXt.h>
#include "cufft_utils.h"
#include "cufft_routines.hpp"
#include "cublas_routines.hpp"
#include "tests.hpp"
#include <complex.h>
// #include "fields.hpp"
// #include "wavevector.hpp"
// #include "common.hpp"
// #include "parameters.hpp"
#include "cuda_kernels.hpp"
#include <argparse/argparse.hpp>
#include "physics_modules.hpp"


// Parameters param;
// int threadsPerBlock = 512;

int main(int argc, char *argv[]) {

    argparse::ArgumentParser program("spooky");

    program.add_argument("--input-dir")
    .help("input directory for cfg file")
    .default_value(std::string("./"));


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

    int success = 1; //fail

    success = test_forward_inverse_transform();
    // test_do_multiplications();
    // test_axpy();

    std::printf("success is %d\n",success);


    return success;
};
