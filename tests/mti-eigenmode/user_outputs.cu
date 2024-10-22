// #include "define_types.hpp"
// // #include "fields.hpp"
// #include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
// #include "fields.hpp"
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

UserOutput::UserOutput() {
    // double lx, ly, lz;
    // read_Parameters();
}

UserOutput::~UserOutput() {
}



void InputOutput::WriteUserTimevarOutput(Fields &fields, Parameters &param, TimeStepping &timestep) {

    NVTX3_FUNC_RANGE();
#ifdef DEBUG
    std::printf("Writing user data output... \n");
#endif

    int blocksPerGrid;
    double t0        = param.t_initial;
    double time_save = timestep.current_time;
    double tend     = param.t_final;
    double output_var = 0.0;

    char data_output_name[64];
    std::sprintf(data_output_name,"timevar-user.spooky");
    std::string fname = param.output_dir + std::string("/data/") + std::string(data_output_name);

    std::ofstream outputfile;
    outputfile.open (fname, std::ios_base::app);
    // outputfile << "Writing this to a file.\n";

    // the first fields.num_fields arrays in tmparray will
    // always contain the real fields for all subsequent operations

    if (param.userOutVar.length > 0){
        for (int i = 0; i < param.userOutVar.length; i++){

            if(!param.userOutVar.name[i].compare(std::string("uservar1"))) {
                output_var = param.userOutVar.customFunction(fields.d_farray[0]);
            }
            else if(!param.userOutVar.name[i].compare(std::string("uservar2"))) {
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


void InputOutput::WriteUserTimevarOutputHeader(Parameters &param) {

#ifdef DEBUG
    std::printf("Writing user data output... \n");
#endif

    char data_output_name[64];
    std::sprintf(data_output_name,"timevar-user.spooky");
    std::string fname = param.output_dir + std::string("/data/") + std::string(data_output_name);

    std::cout << "user output file name: " << data_output_name << std::endl;

    std::ofstream outputfile;
    outputfile.open (fname, std::ios_base::app);


    outputfile << "## This file contains the time evolution of the following quantities: \n";
    outputfile << "## \t";

    // for(int i = 0 ; i < param.spookyOutVar.length ; i++) {
    //     outputfile << param.spookyOutVar.name[i]  << "\t";
    // }

    if (param.userOutVar.length > 0){
        for (int i = 0; i < param.userOutVar.length; i++){
            outputfile << param.userOutVar.name[i]  << "\t";
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
