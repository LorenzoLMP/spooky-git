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

    int nbins = supervisor_ptr->fields_ptr->wavevector.nbins;

    CUDA_RT_CALL(cudaMalloc(&d_output_spectrum, (size_t) sizeof(scalar_type) * nbins);



}

InputOutput::~InputOutput(){

    CUDA_RT_CALL(cudaFree(d_output_spectrum));
    // delete timevar_timer;
    // delete datadump_timer;

}
