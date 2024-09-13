#include "define_types.hpp"
#include "cufft_routines.hpp"
#include "cuda_kernels_generic.hpp"
#include "spooky.hpp"
#include "common.hpp"
// #include "../libs/HighFive/include/highfive/highfive.hpp"
// #include <highfive/highfive.hpp>
#include "fields.hpp"
// #include "hdf5_io.hpp"
// #include "output_timevar.hpp"
// #include "spooky_outputs.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

void Fields::write_data_output() {

    NVTX3_FUNC_RANGE();
#ifdef DEBUG
    std::printf("Writing data output... \n");
#endif

    int blocksPerGrid;
    double t0        = param->t_initial;
    double time_save = current_time;
    double tend     = param->t_final;
    double output_var = 0.0;

    char data_output_name[16];
    std::sprintf(data_output_name,"timevar.spooky");
    std::string fname = param->output_dir + std::string("/data/") + std::string(data_output_name);

    std::ofstream outputfile;
    outputfile.open (fname, std::ios_base::app);
    // outputfile << "Writing this to a file.\n";

    // assign fields to [num_fields] tmparray (memory block starts at d_all_tmparray)
    blocksPerGrid = ( num_fields * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    ComplexVecAssign<<<blocksPerGrid, threadsPerBlock>>>((cufftDoubleComplex *)d_all_fields, (cufftDoubleComplex *)d_all_tmparray, num_fields * ntotal_complex);

    // compute FFTs from complex to real fields
    // the first num_fields arrays in tmparray will
    // always contain the real fields for all subsequent operations
    for (int n = 0; n < num_fields; n++){
        c2r_fft(d_tmparray[n], d_tmparray_r[n]);
    }

    // std::printf("length timevar array = %d \n",param->spookyOutVar.length);
    // begin loop through the output variables
    for (int i = 0; i < param->spookyOutVar.length; i++){
        // std::printf("variable = %s \t",param->spookyOutVar.name[i]);
        // std::cout << param->spookyOutVar.name[i] << "\t";
        // if(!strcmp(param->spookyOutVar.name[i],"t")) {
        if(!param->spookyOutVar.name[i].compare(std::string("t"))) {
            // current time
            output_var = current_time;
        }
        // else if(!param->spookyOutVar.name[i].compare(std::string("ev"))) {
        //     // kinetic energy
        //     output_var = param->spookyOutVar.computeEnergy((data_type *) d_all_tmparray);
        // }
        else {
            output_var = -1.0;
        }

        outputfile << std::scientific << std::setprecision(8) << output_var << "\t";
    }

    outputfile << "\n";
    outputfile.close();
}
