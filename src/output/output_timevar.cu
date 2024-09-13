#include "../define_types.hpp"
// #include "fields.hpp"
// #include "cufft_routines.hpp"
#include "../spooky.hpp"
#include "../common.hpp"
// #include "../libs/HighFive/include/highfive/highfive.hpp"
// #include <highfive/highfive.hpp>
#include "../fields.hpp"
// #include "hdf5_io.hpp"
// #include "output_timevar.hpp"
#include "spooky_outputs.hpp"
#include <iostream>
#include <fstream>

void Fields::write_data_output() {

    NVTX3_FUNC_RANGE();
#ifdef DEBUG
    std::printf("Writing data output... \n");
#endif

    double t0        = param->t_initial;
    double time_save = current_time;
    double tend     = param->t_final;

    char data_output_name[16];
    std::sprintf(data_output_name,"timevar.spooky");
    std::string fname = param->output_dir + std::string("/data/") + std::string(data_output_name);

    ofstream outputfile;
    outputfile.open (fname);
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

    // begin loop through the output variables
    for (int i = 0; i < param->spookyOutVar.length; i++){

        if(!strcmp(param->spookyOutVar.name[i],"t")) {
            // current time
            output_var = t;
        }
        // else if(!strcmp(param->spookyOutVar.name[i],"ev")) {
        //     // kinetic energy
        //     output_var = param->spookyOutVar.computeEnergy();
        // }

        outputfile << std::scientific << std::setprecision(8) << output_var << "\t";
    }

    outputfile << "\n";
    outputfile.close();
}
