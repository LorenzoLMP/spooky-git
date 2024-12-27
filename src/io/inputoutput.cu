#include "common.hpp"
#include "fields.hpp"
// #include "cufft_routines.hpp"
// #include "cublas_routines.hpp"
// #include "cuda_kernels.hpp"
#include "parameters.hpp"
#include "inputoutput.hpp"
#include "timestepping.hpp"
// #include "supervisor.hpp"

InputOutput::InputOutput(Supervisor &sup_in) {

    supervisor_ptr = &sup_in;

    // timevar_timer = new Timer();
    // datadump_timer = new Timer();

    t_lastsnap = 0.0;
    t_lastvar = 0.0;
    num_save = 0;
}

InputOutput::~InputOutput(){
    // delete timevar_timer;
    // delete datadump_timer;

}
