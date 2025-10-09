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


void InputOutput::WriteUserTimevarOutput() {

    // NVTX3_FUNC_RANGE();

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;
    std::shared_ptr<TimeStepping> timestep_ptr = supervisor_ptr->timestep_ptr;

    if (param_ptr->debug > 0) {
        std::printf("Writing user data output... \n");
    }

    // int blocksPerGrid;
    double t0        = param_ptr->t_initial;
    double time_save = timestep_ptr->current_time;
    double tend     = param_ptr->t_final;
    double output_var = 0.0;

    char data_output_name[16];
    std::sprintf(data_output_name,"user-timevar.spooky");
    std::string fname = param_ptr->output_dir + std::string("/data/") + std::string(data_output_name);

    std::ofstream outputfile;
    outputfile.open (fname, std::ios_base::app);
    // outputfile << "Writing this to a file.\n";

    // the first fields_ptr->num_fields arrays in tmparray will
    // always contain the real fields for all subsequent operations

    // if (param_ptr->userOutVar.length_timevar > 0){
    for (int i = 0; i < param_ptr->userOutVar.length_timevar; i++){

        output_var = -1.0;
        
        if (param_ptr->anisotropic_diffusion) {

            if(!param_ptr->userOutVar.name_timevar[i].compare(std::string("kpartheta"))) {
                // parallel temperature length
                output_var = param_ptr->userOutVar.computekpartheta(fields_ptr->d_all_fields,
                fields_ptr->d_all_buffer_r);
            }
        }

        if(!param_ptr->userOutVar.name_timevar[i].compare(std::string("uservar1"))) {
            output_var = param_ptr->userOutVar.customFunction(fields_ptr->d_farray[0]);
        }
        else if(!param_ptr->userOutVar.name_timevar[i].compare(std::string("uservar2"))) {
            output_var = 0.0;
        }
        
        outputfile << std::scientific << std::setprecision(8) << output_var << "\t";
    }
    // }

    outputfile << "\n";
    outputfile.close();
}


void InputOutput::WriteUserTimevarOutputHeader() {

    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    if (param_ptr->debug > 0) {
        std::printf("Writing user data header... \n");
    }


    char data_output_name[16];
    std::sprintf(data_output_name,"user-timevar.spooky");
    std::string fname = param_ptr->output_dir + std::string("/data/") + std::string(data_output_name);

    std::ofstream outputfile;
    outputfile.open (fname, std::ios_base::app);


    outputfile << "## This file contains the time evolution of the following quantities: \n";
    outputfile << "## \t";

    // if (param_ptr->userOutVar.length_timevar > 0){
    for (int i = 0; i < param_ptr->userOutVar.length_timevar; i++){
        outputfile << param_ptr->userOutVar.name_timevar[i]  << "\t";
    }
    // }

    outputfile << "\n";
    outputfile.close();
}
