#include "main.hpp"
#include <cuda_runtime.h>
// #include <cufftXt.h>
#include "cufft_utils.h"
#include "cufft_routines.hpp"
#include "cublas_routines.hpp"
#include "tests.hpp"
#include <complex.h>
// #include "fields.hpp"
// #include "wavevector.hpp"
#include "common.hpp"
// #include "parameters.hpp"
#include "cuda_kernels.hpp"
#include <argparse/argparse.hpp>
// #include "physics_modules.hpp"


#define SET 0
#define ADD 1
// Parameters param;
int threadsPerBlock = 512;
// int threadsPerBlock{512};
Variables vars;
Grid grid;

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

 //    vars.NUM_FIELDS = 0;
	// vars.VX = 0; vars.VY = 0; vars.VZ = 0;
 //    vars.BX = 0; vars.BY = 0; vars.BZ = 0;
 //    vars.TH = 0;
 //    vars.VEL = 0; vars.MAG = 0;
 //
	// vars.KX = 0;
	// vars.KY = 1;
	// vars.KZ = 2;
 //
 //
 //
	grid.NX = (size_t) 512;
	grid.NY = (size_t) 512;
	grid.NZ = (size_t) 512;

	grid.FFT_SIZE[0] = grid.NX;
	grid.FFT_SIZE[1] = grid.NY;
	grid.FFT_SIZE[2] = grid.NZ;

	grid.NTOTAL = grid.NX * grid.NY * grid.NZ;

	grid.NTOTAL_COMPLEX = grid.NX * grid.NY * (( grid.NZ / 2) + 1);
 //
    success = test_forward_inverse_transform();
    // test_do_multiplications();
    // test_axpy();

    std::printf("success is %d\n",success);


    return success;
};
