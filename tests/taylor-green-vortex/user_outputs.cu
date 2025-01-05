#include "common.hpp"
// #include "parameters.hpp"
// #include "user_outputs.hpp"
#include "cublas_routines.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "cufft_routines.hpp"
#include "user_defined_cuda_kernels.hpp"

#include "fields.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include "parameters.hpp" //includes user_outputs
#include "inputoutput.hpp"
#include "timestepping.hpp"
#include "supervisor.hpp"

UserOutput::UserOutput(Supervisor &sup_in)
    : SpookyOutput(sup_in) {
    // double lx, ly, lz;
    // read_Parameters();
    // supervisor_ptr = &sup_in;
}

// UserOutput::~UserOutput() {
// }



void InputOutput::WriteUserTimevarOutput() {

    // NVTX3_FUNC_RANGE();

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;
    std::shared_ptr<TimeStepping> timestep_ptr = supervisor_ptr->timestep_ptr;

#ifdef DEBUG
    std::printf("Writing user data output... \n");
#endif

    int blocksPerGrid;
    double t0        = param_ptr->t_initial;
    double time_save = timestep_ptr->current_time;
    double tend     = param_ptr->t_final;
    double output_var = 0.0;

    char data_output_name[16];
    std::sprintf(data_output_name,"timevar-user.spooky");
    std::string fname = param_ptr->output_dir + std::string("/data/") + std::string(data_output_name);

    std::ofstream outputfile;
    outputfile.open (fname, std::ios_base::app);
    // outputfile << "Writing this to a file.\n";

    // the first fields_ptr->num_fields arrays in tmparray will
    // always contain the real fields for all subsequent operations

    if (param_ptr->userOutVar.length > 0){
        for (int i = 0; i < param_ptr->userOutVar.length; i++){

            if(!param_ptr->userOutVar.name[i].compare(std::string("uservar1"))) {
                output_var = param_ptr->userOutVar.customFunction(fields_ptr->d_farray[0]);
            }
            else if(!param_ptr->userOutVar.name[i].compare(std::string("uservar2"))) {
                output_var = 0.0;
            }
            else {
                output_var = -1.0;
            }
            outputfile << std::scientific << std::setprecision(8) << output_var << "\t";
        }
    }

    outputfile << "\n";
    outputfile.close();
}


void InputOutput::WriteUserTimevarOutputHeader() {

#ifdef DEBUG
    std::printf("Writing data output... \n");
#endif

     std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    char data_output_name[16];
    std::sprintf(data_output_name,"timevar-user.spooky");
    std::string fname = param_ptr->output_dir + std::string("/data/") + std::string(data_output_name);

    std::ofstream outputfile;
    outputfile.open (fname, std::ios_base::app);


    outputfile << "## This file contains the time evolution of the following quantities: \n";
    outputfile << "## \t";

    // for(int i = 0 ; i < param_ptr->spookyOutVar.length ; i++) {
    //     outputfile << param_ptr->spookyOutVar.name[i]  << "\t";
    // }

    if (param_ptr->userOutVar.length > 0){
        for (int i = 0; i < param_ptr->userOutVar.length; i++){
            outputfile << param_ptr->userOutVar.name[i]  << "\t";
        }
    }

    outputfile << "\n";
    outputfile.close();
}

scalar_type UserOutput::customFunction( data_type *vcomplex ) {
    /***
     * This function uses complex input to compute the "energy"
     * The modes with k>0 only have half the energy (because the k<0 is not present).
     * Here we multiply all k modes by 2 and then subtract once the energy in the k=0 mode.
     * The total is then divided by 2 to give quantity (i.e. Energy ~ (1/2) v^2)
     ***/

    // cublasStatus_t stat;
    // scalar_type norm = 0.0;
    scalar_type var = 0.0;
    // scalar_type subtract = 0.0;
    // scalar_type tmp = 0.0;

    return var;
}
