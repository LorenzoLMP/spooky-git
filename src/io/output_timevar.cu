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
#include "parameters.hpp"
#include "inputoutput.hpp"
#include "timestepping.hpp"

void InputOutput::write_data_output(Fields &fields, Parameters &param, TimeStepping &timestep) {

    NVTX3_FUNC_RANGE();
#ifdef DEBUG
    std::printf("Writing data output... \n");
#endif

    int blocksPerGrid;
    double t0        = param.t_initial;
    double time_save = timestep.current_time;
    double tend     = param.t_final;
    double output_var = 0.0;

    char data_output_name[16];
    std::sprintf(data_output_name,"timevar.spooky");
    std::string fname = param.output_dir + std::string("/data/") + std::string(data_output_name);

    std::ofstream outputfile;
    outputfile.open (fname, std::ios_base::app);
    // outputfile << "Writing this to a file.\n";

    // assign fields to [fields.num_fields] tmparray (memory block starts at d_all_tmparray)
    blocksPerGrid = ( fields.num_fields * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    ComplexVecAssign<<<blocksPerGrid, threadsPerBlock>>>((cufftDoubleComplex *)fields.d_all_fields, (cufftDoubleComplex *)fields.d_all_tmparray, fields.num_fields * ntotal_complex);

    // compute FFTs from complex to real fields
    // the first fields.num_fields arrays in tmparray will
    // always contain the real fields for all subsequent operations
    for (int n = 0; n < fields.num_fields; n++){
        c2r_fft(fields.d_tmparray[n], fields.d_tmparray_r[n]);
    }

    // std::printf("length timevar array = %d \n",param.spookyOutVar.length);
    // begin loop through the output variables
    for (int i = 0; i < param.spookyOutVar.length; i++){
        // std::printf("variable = %s \t",param.spookyOutVar.name[i]);
        // std::cout << param.spookyOutVar.name[i] << "\t";
        // if(!strcmp(param.spookyOutVar.name[i],"t")) {
        if(!param.spookyOutVar.name[i].compare(std::string("t"))) {
            // current time
            output_var = timestep.current_time;
        }
#ifdef INCOMPRESSIBLE
        else if(!param.spookyOutVar.name[i].compare(std::string("ev"))) {
            // kinetic energy
            output_var = param.spookyOutVar.computeEnergy(fields.d_farray[VX]);
            output_var += param.spookyOutVar.computeEnergy(fields.d_farray[VY]);
            output_var += param.spookyOutVar.computeEnergy(fields.d_farray[VZ]);
        }
        else if(!param.spookyOutVar.name[i].compare(std::string("Kx"))) {
            // kinetic energy x
            output_var = param.spookyOutVar.computeEnergy(fields.d_farray[VX]);
        }
        else if(!param.spookyOutVar.name[i].compare(std::string("Ky"))) {
            // kinetic energy y
            output_var = param.spookyOutVar.computeEnergy(fields.d_farray[VY]);
        }
        else if(!param.spookyOutVar.name[i].compare(std::string("Kz"))) {
            // kinetic energy z
            output_var = param.spookyOutVar.computeEnergy(fields.d_farray[VZ]);
        }
        else if(!param.spookyOutVar.name[i].compare(std::string("vxvy"))) {
            // reynolds stresses
            output_var = param.spookyOutVar.twoFieldCorrelation(fields.d_tmparray_r[VX], fields.d_tmparray_r[VY]);
        }
        else if(!param.spookyOutVar.name[i].compare(std::string("vyvz"))) {
            // reynolds stresses
            output_var = param.spookyOutVar.twoFieldCorrelation(fields.d_tmparray_r[VY], fields.d_tmparray_r[VZ]);
        }
        else if(!param.spookyOutVar.name[i].compare(std::string("vyvz"))) {
            // reynolds stresses
            output_var = param.spookyOutVar.twoFieldCorrelation(fields.d_tmparray_r[VY], fields.d_tmparray_r[VZ]);
        }
        else if(!param.spookyOutVar.name[i].compare(std::string("w2"))) {
            // enstrophy
            output_var = param.spookyOutVar.computeEnstrophy(fields.d_all_fields + ntotal_complex * VX,
                                             fields.wavevector.d_all_kvec,
                                             fields.d_all_tmparray + ntotal_complex * fields.num_fields);
        }
#endif
#ifdef MHD
        else if(!param.spookyOutVar.name[i].compare(std::string("em"))) {
            // magnetic energy
            output_var = param.spookyOutVar.computeEnergy(fields.d_farray[BX]);
            output_var += param.spookyOutVar.computeEnergy(fields.d_farray[BY]);
            output_var += param.spookyOutVar.computeEnergy(fields.d_farray[BZ]);
        }
        else if(!param.spookyOutVar.name[i].compare(std::string("Mx"))) {
            // magnetic energy x
            output_var = param.spookyOutVar.computeEnergy(fields.d_farray[BX]);
        }
        else if(!param.spookyOutVar.name[i].compare(std::string("My"))) {
            // magnetic energy y
            output_var = param.spookyOutVar.computeEnergy(fields.d_farray[BY]);
        }
        else if(!param.spookyOutVar.name[i].compare(std::string("Mz"))) {
            // magnetic energy z
            output_var = param.spookyOutVar.computeEnergy(fields.d_farray[BZ]);
        }
        else if(!param.spookyOutVar.name[i].compare(std::string("bxby"))) {
            // maxwell stresses
            output_var = param.spookyOutVar.twoFieldCorrelation(fields.d_tmparray_r[BX], fields.d_tmparray_r[BY]);
        }
        else if(!param.spookyOutVar.name[i].compare(std::string("bybz"))) {
            // maxwell stresses
            output_var = param.spookyOutVar.twoFieldCorrelation(fields.d_tmparray_r[BY], fields.d_tmparray_r[BZ]);
        }
        else if(!param.spookyOutVar.name[i].compare(std::string("bybz"))) {
            // maxwell stresses
            output_var = param.spookyOutVar.twoFieldCorrelation(fields.d_tmparray_r[BY], fields.d_tmparray_r[BZ]);
        }
        else if(!param.spookyOutVar.name[i].compare(std::string("j2"))) {
            // compute total current rms
            // we can reuse the computeEnstrophy function
            output_var = param.spookyOutVar.computeEnstrophy(fields.d_all_fields + ntotal_complex * BX,
                                             fields.wavevector.d_all_kvec,
                                             fields.d_all_tmparray + ntotal_complex * fields.num_fields);
        }
#endif
#if defined(BOUSSINESQ) || defined(HEAT_EQ)
        else if(!param.spookyOutVar.name[i].compare(std::string("et"))) {
            // thermal/potential energy
            output_var = param.spookyOutVar.computeEnergy(fields.d_farray[TH]);
        }
        else if(!param.spookyOutVar.name[i].compare(std::string("dissgradT"))) {
            // thermal dissipation (isotropic or anisotropic)
            #if defined(ANISOTROPIC_DIFFUSION) && defined(MHD)
            // the minus sign is for consistency with snoopy
            output_var = - param.spookyOutVar.computeAnisoDissipation(fields.wavevector.d_all_kvec,
                                                fields.d_all_fields,
                                                fields.d_farray,
                                                fields.d_farray_r,
                                                fields.d_all_tmparray,
                                                fields.d_tmparray,
                                                fields.d_tmparray_r,
                                                fields.wavevector.d_mask,
                                                fields.num_fields);
            // output_var = param.spookyOutVar.computeAnisoDissipation(d_all_fields,
            //                                     wavevector.d_all_kvec,
            //                                     (scalar_type *)d_all_tmparray,
            //                                     d_all_tmparray + ntotal_complex * fields.num_fields);
            #else
            output_var = param.spookyOutVar.computeDissipation(fields.d_farray[TH],
                                                fields.wavevector.d_all_kvec,
                                                fields.d_all_tmparray + ntotal_complex * fields.num_fields);
            #endif
        }
        #if defined(ANISOTROPIC_DIFFUSION) && defined(MHD)
        else if(!param.spookyOutVar.name[i].compare(std::string("fluxbbgradT"))) {
            // thermal dissipation (isotropic or anisotropic)

            output_var = param.spookyOutVar.computeAnisoInjection(fields.wavevector.d_all_kvec,
                                                fields.d_all_fields,
                                                fields.d_farray,
                                                fields.d_farray_r,
                                                fields.d_all_tmparray,
                                                fields.d_tmparray,
                                                fields.d_tmparray_r,
                                                fields.wavevector.d_mask,
                                                fields.num_fields);
        }
        #endif
#endif
#if defined(BOUSSINESQ) && defined(INCOMPRESSIBLE)
        else if(!param.spookyOutVar.name[i].compare(std::string("thvx"))) {
            // convective flux
            output_var = param.spookyOutVar.twoFieldCorrelation(fields.d_tmparray_r[TH], fields.d_tmparray_r[VX]);
        }
        else if(!param.spookyOutVar.name[i].compare(std::string("thvz"))) {
            // convective flux
            output_var = param.spookyOutVar.twoFieldCorrelation(fields.d_tmparray_r[TH], fields.d_tmparray_r[VZ]);
        }
#endif
        else {
            output_var = -1.0;
        }

        outputfile << std::scientific << std::setprecision(8) << output_var << "\t";
    }

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


void InputOutput::write_data_output_header(Parameters &param) {

#ifdef DEBUG
    std::printf("Writing data output... \n");
#endif

    char data_output_name[16];
    std::sprintf(data_output_name,"timevar.spooky");
    std::string fname = param.output_dir + std::string("/data/") + std::string(data_output_name);

    std::ofstream outputfile;
    outputfile.open (fname, std::ios_base::app);


    outputfile << "## This file contains the time evolution of the following quantities: \n";
    outputfile << "## \t";

    for(int i = 0 ; i < param.spookyOutVar.length ; i++) {
        outputfile << param.spookyOutVar.name[i]  << "\t";
    }

    if (param.userOutVar.length > 0){
        for (int i = 0; i < param.userOutVar.length; i++){
            outputfile << param.userOutVar.name[i]  << "\t";
        }
    }

    outputfile << "\n";
    outputfile.close();
}
