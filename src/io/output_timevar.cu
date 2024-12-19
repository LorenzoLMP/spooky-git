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
#include "supervisor.hpp"

void InputOutput::WriteTimevarOutput() {

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;
    std::shared_ptr<TimeStepping> timestep_ptr = supervisor_ptr->timestep_ptr;

    NVTX3_FUNC_RANGE();
#ifdef DEBUG
    std::printf("Writing data output... \n");
#endif

    int blocksPerGrid;
    double t0        = param_ptr->t_initial;
    double time_save = timestep_ptr->current_time;
    double tend     = param_ptr->t_final;
    double output_var = 0.0;

    char data_output_name[16];
    std::sprintf(data_output_name,"timevar.spooky");
    std::string fname = param_ptr->output_dir + std::string("/data/") + std::string(data_output_name);

    std::ofstream outputfile;
    outputfile.open (fname, std::ios_base::app);
    // outputfile << "Writing this to a file.\n";

    // // assign fields to [fields_ptr->num_fields] tmparray (memory block starts at d_all_tmparray)
    // blocksPerGrid = ( fields_ptr->num_fields * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    // ComplexVecAssign<<<blocksPerGrid, threadsPerBlock>>>((data_type *)fields_ptr->d_all_fields, (data_type *)fields_ptr->d_all_tmparray, fields_ptr->num_fields * ntotal_complex);
    //
    // // compute FFTs from complex to real fields
    // // the first fields_ptr->num_fields arrays in tmparray will
    // // always contain the real fields for all subsequent operations
    // for (int n = 0; n < fields_ptr->num_fields; n++){
    //     c2r_fft(fields_ptr->d_tmparray[n], fields_ptr->d_farray_buffer_r[n]);
    // }

    supervisor_ptr->Complex2RealFields(fields_ptr->d_all_fields, fields_ptr->d_all_buffer_r, fields_ptr->num_fields)

    // std::printf("length timevar array = %d \n",param_ptr->spookyOutVar.length);
    // begin loop through the output variables
    for (int i = 0; i < param_ptr->spookyOutVar.length; i++){
        // std::printf("variable = %s \t",param_ptr->spookyOutVar.name[i]);
        // std::cout << param_ptr->spookyOutVar.name[i] << "\t";
        // if(!strcmp(param_ptr->spookyOutVar.name[i],"t")) {
        if(!param_ptr->spookyOutVar.name[i].compare(std::string("t"))) {
            // current time
            output_var = timestep_ptr->current_time;
        }
#ifdef INCOMPRESSIBLE
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("ev"))) {
            // kinetic energy
            output_var = param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[VX]);
            output_var += param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[VY]);
            output_var += param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[VZ]);
        }
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("Kx"))) {
            // kinetic energy x
            output_var = param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[VX]);
        }
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("Ky"))) {
            // kinetic energy y
            output_var = param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[VY]);
        }
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("Kz"))) {
            // kinetic energy z
            output_var = param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[VZ]);
        }
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("vxvy"))) {
            // reynolds stresses
            output_var = param_ptr->spookyOutVar.twoFieldCorrelation(fields_ptr->d_farray_buffer_r[VX], fields_ptr->d_farray_buffer_r[VY]);
        }
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("vyvz"))) {
            // reynolds stresses
            output_var = param_ptr->spookyOutVar.twoFieldCorrelation(fields_ptr->d_farray_buffer_r[VY], fields_ptr->d_farray_buffer_r[VZ]);
        }
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("vyvz"))) {
            // reynolds stresses
            output_var = param_ptr->spookyOutVar.twoFieldCorrelation(fields_ptr->d_farray_buffer_r[VY], fields_ptr->d_farray_buffer_r[VZ]);
        }
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("w2"))) {
            // enstrophy
            output_var = param_ptr->spookyOutVar.computeEnstrophy(fields_ptr->d_farray[VX],
                                                                  fields_ptr->d_farray[VY],
                                                                  fields_ptr->d_farray[VZ]);
        }
#endif
#ifdef MHD
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("em"))) {
            // magnetic energy
            output_var = param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[BX]);
            output_var += param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[BY]);
            output_var += param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[BZ]);
        }
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("Mx"))) {
            // magnetic energy x
            output_var = param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[BX]);
        }
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("My"))) {
            // magnetic energy y
            output_var = param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[BY]);
        }
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("Mz"))) {
            // magnetic energy z
            output_var = param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[BZ]);
        }
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("bxby"))) {
            // maxwell stresses
            output_var = param_ptr->spookyOutVar.twoFieldCorrelation(fields_ptr->d_farray_buffer_r[BX], fields_ptr->d_farray_buffer_r[BY]);
        }
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("bybz"))) {
            // maxwell stresses
            output_var = param_ptr->spookyOutVar.twoFieldCorrelation(fields_ptr->d_farray_buffer_r[BY], fields_ptr->d_farray_buffer_r[BZ]);
        }
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("bybz"))) {
            // maxwell stresses
            output_var = param_ptr->spookyOutVar.twoFieldCorrelation(fields_ptr->d_farray_buffer_r[BY], fields_ptr->d_farray_buffer_r[BZ]);
        }
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("j2"))) {
            // compute total current rms
            // we can reuse the computeEnstrophy function
            output_var = param_ptr->spookyOutVar.computeEnstrophy(fields_ptr->d_farray[BX],
                                                                  fields_ptr->d_farray[BY],
                                                                  fields_ptr->d_farray[BZ]);
        }
#endif
#if defined(BOUSSINESQ) || defined(HEAT_EQ)
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("et"))) {
            // thermal/potential energy
            output_var = param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[TH]);
        }
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("dissgradT"))) {
            // thermal dissipation (isotropic or anisotropic)
            #if defined(ANISOTROPIC_DIFFUSION) && defined(MHD)
            // the minus sign is for consistency with snoopy
            output_var = - param_ptr->spookyOutVar.computeAnisoDissipation(fields_ptr->d_all_fields,
                                                                           fields_ptr->d_all_buffer_r);
            // output_var = param_ptr->spookyOutVar.computeAnisoDissipation(d_all_fields,
            //                                     wavevector.d_all_kvec,
            //                                     (scalar_type *)d_all_tmparray,
            //                                     d_all_tmparray + ntotal_complex * fields_ptr->num_fields);
            #else
            output_var = param_ptr->spookyOutVar.computeDissipation(fields_ptr->d_farray[TH]);
            #endif
        }
        #if defined(ANISOTROPIC_DIFFUSION) && defined(MHD)
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("fluxbbgradT"))) {
            // thermal dissipation (isotropic or anisotropic)
            // for now it's only anisotropic
            output_var = param_ptr->spookyOutVar.computeAnisoInjection(fields_ptr->d_all_fields,
                                                                        fields_ptr->d_all_buffer_r);
        }
        #endif
#endif
#if defined(BOUSSINESQ) && defined(INCOMPRESSIBLE)
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("thvx"))) {
            // convective flux
            output_var = param_ptr->spookyOutVar.twoFieldCorrelation(fields_ptr->d_farray_buffer_r[TH], fields_ptr->d_farray_buffer_r[VX]);
        }
        else if(!param_ptr->spookyOutVar.name[i].compare(std::string("thvz"))) {
            // convective flux
            output_var = param_ptr->spookyOutVar.twoFieldCorrelation(fields_ptr->d_farray_buffer_r[TH], fields_ptr->d_farray_buffer_r[VZ]);
        }
#endif
        else {
            output_var = -1.0;
        }

        outputfile << std::scientific << std::setprecision(8) << output_var << "\t";
    }

    // if (param_ptr->userOutVar.length > 0){
    //     for (int i = 0; i < param_ptr->userOutVar.length; i++){
    //
    //         if(!param_ptr->userOutVar.name[i].compare(std::string("uservar1"))) {
    //             output_var = param_ptr->userOutVar.customFunction(fields_ptr->d_farray[0]);
    //         }
    //         else if(!param_ptr->userOutVar.name[i].compare(std::string("uservar2"))) {
    //             output_var = 0.0;
    //         }
    //         else {
    //             output_var = -1.0;
    //         }
    //         outputfile << std::scientific << std::setprecision(8) << output_var << "\t";
    //     }
    // }

    outputfile << "\n";
    outputfile.close();
}


void InputOutput::WriteTimevarOutputHeader() {

    std::shared_ptr<Parameters> param = supervisor_ptr->param;

#ifdef DEBUG
    std::printf("Writing data output... \n");
#endif

    char data_output_name[16];
    std::sprintf(data_output_name,"timevar.spooky");
    std::string fname = param_ptr->output_dir + std::string("/data/") + std::string(data_output_name);

    std::ofstream outputfile;
    outputfile.open (fname, std::ios_base::app);


    outputfile << "## This file contains the time evolution of the following quantities: \n";
    outputfile << "## \t";

    for(int i = 0 ; i < param_ptr->spookyOutVar.length ; i++) {
        outputfile << param_ptr->spookyOutVar.name[i]  << "\t";
    }

    // if (param_ptr->userOutVar.length > 0){
    //     for (int i = 0; i < param_ptr->userOutVar.length; i++){
    //         outputfile << param_ptr->userOutVar.name[i]  << "\t";
    //     }
    // }

    outputfile << "\n";
    outputfile.close();
}
