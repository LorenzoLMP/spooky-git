#include "define_types.hpp"
#include "fields.hpp"
// #include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
// #include "cublas_routines.hpp"
// #include "cuda_kernels.hpp"
#include "parameters.hpp"
#include "inputoutput.hpp"
#include "timestepping.hpp"

InputOutput::InputOutput() {

    t_lastsnap = 0.0;
    t_lastvar = 0.0;
    num_save = 0;
}

InputOutput::~InputOutput(){

}
