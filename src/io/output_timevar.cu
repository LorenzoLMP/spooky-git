#include "common.hpp"
#include "cufft_routines.hpp"
#include "cuda_kernels_generic.hpp"
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

    if (param_ptr->debug > 0) {
        std::printf("Writing data output... \n");
    }

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

    // // assign fields to [vars.NUM_FIELDS] tmparray (memory block starts at d_all_tmparray)
    // blocksPerGrid = ( vars.NUM_FIELDS * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
    // ComplexVecAssign<<<blocksPerGrid, threadsPerBlock>>>((data_type *)fields_ptr->d_all_fields, (data_type *)fields_ptr->d_all_tmparray, vars.NUM_FIELDS * grid.NTOTAL_COMPLEX);
    //
    // // compute FFTs from complex to real fields
    // // the first vars.NUM_FIELDS arrays in tmparray will
    // // always contain the real fields for all subsequent operations
    // for (int n = 0; n < vars.NUM_FIELDS; n++){
    //     c2r_fft(fields_ptr->d_tmparray[n], fields_ptr->d_farray_buffer_r[n]);
    // }

    supervisor_ptr->Complex2RealFields(fields_ptr->d_all_fields, fields_ptr->d_all_buffer_r, vars.NUM_FIELDS);

    // std::printf("length timevar array = %d \n",param_ptr->spookyOutVar.length);
    // begin loop through the output variables
    for (int i = 0; i < param_ptr->spookyOutVar.length; i++){
        // std::printf("variable = %s \t",param_ptr->spookyOutVar.name[i]);
        // std::cout << param_ptr->spookyOutVar.name[i] << "\t";
        // if(!strcmp(param_ptr->spookyOutVar.name[i],"t")) {
        output_var = -1.0;

        if(!param_ptr->spookyOutVar.name[i].compare(std::string("t"))) {
            // current time
            output_var = timestep_ptr->current_time;
        }

        if (param_ptr->incompressible) {

            if(!param_ptr->spookyOutVar.name[i].compare(std::string("ev"))) {
                // kinetic energy
                output_var = param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[vars.VX]);
                output_var += param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[vars.VY]);
                output_var += param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[vars.VZ]);
            }
            else if(!param_ptr->spookyOutVar.name[i].compare(std::string("Kx"))) {
                // kinetic energy x
                output_var = param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[vars.VX]);
            }
            else if(!param_ptr->spookyOutVar.name[i].compare(std::string("Ky"))) {
                // kinetic energy y
                output_var = param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[vars.VY]);
            }
            else if(!param_ptr->spookyOutVar.name[i].compare(std::string("Kz"))) {
                // kinetic energy z
                output_var = param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[vars.VZ]);
            }
            else if(!param_ptr->spookyOutVar.name[i].compare(std::string("vxvy"))) {
                // reynolds stresses
                output_var = param_ptr->spookyOutVar.twoFieldCorrelation(fields_ptr->d_farray_buffer_r[vars.VX], fields_ptr->d_farray_buffer_r[vars.VY]);
            }
            else if(!param_ptr->spookyOutVar.name[i].compare(std::string("vyvz"))) {
                // reynolds stresses
                output_var = param_ptr->spookyOutVar.twoFieldCorrelation(fields_ptr->d_farray_buffer_r[vars.VY], fields_ptr->d_farray_buffer_r[vars.VZ]);
            }
            else if(!param_ptr->spookyOutVar.name[i].compare(std::string("vyvz"))) {
                // reynolds stresses
                output_var = param_ptr->spookyOutVar.twoFieldCorrelation(fields_ptr->d_farray_buffer_r[vars.VY], fields_ptr->d_farray_buffer_r[vars.VZ]);
            }
            else if(!param_ptr->spookyOutVar.name[i].compare(std::string("w2"))) {
                // enstrophy
                output_var = param_ptr->spookyOutVar.computeEnstrophy(fields_ptr->d_farray[vars.VX],
                                                                    fields_ptr->d_farray[vars.VY],
                                                                    fields_ptr->d_farray[vars.VZ]);
            }
        }
        if (param_ptr->mhd) {

            if(!param_ptr->spookyOutVar.name[i].compare(std::string("em"))) {
                // magnetic energy
                output_var = param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[vars.BX]);
                output_var += param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[vars.BY]);
                output_var += param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[vars.BZ]);
            }
            else if(!param_ptr->spookyOutVar.name[i].compare(std::string("Mx"))) {
                // magnetic energy x
                output_var = param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[vars.BX]);
            }
            else if(!param_ptr->spookyOutVar.name[i].compare(std::string("My"))) {
                // magnetic energy y
                output_var = param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[vars.BY]);
            }
            else if(!param_ptr->spookyOutVar.name[i].compare(std::string("Mz"))) {
                // magnetic energy z
                output_var = param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[vars.BZ]);
            }
            else if(!param_ptr->spookyOutVar.name[i].compare(std::string("bxby"))) {
                // maxwell stresses
                output_var = param_ptr->spookyOutVar.twoFieldCorrelation(fields_ptr->d_farray_buffer_r[vars.BX], fields_ptr->d_farray_buffer_r[vars.BY]);
            }
            else if(!param_ptr->spookyOutVar.name[i].compare(std::string("bybz"))) {
                // maxwell stresses
                output_var = param_ptr->spookyOutVar.twoFieldCorrelation(fields_ptr->d_farray_buffer_r[vars.BY], fields_ptr->d_farray_buffer_r[vars.BZ]);
            }
            else if(!param_ptr->spookyOutVar.name[i].compare(std::string("bybz"))) {
                // maxwell stresses
                output_var = param_ptr->spookyOutVar.twoFieldCorrelation(fields_ptr->d_farray_buffer_r[vars.BY], fields_ptr->d_farray_buffer_r[vars.BZ]);
            }
            else if(!param_ptr->spookyOutVar.name[i].compare(std::string("j2"))) {
                // compute total current rms
                // we can reuse the computeEnstrophy function
                output_var = param_ptr->spookyOutVar.computeEnstrophy(fields_ptr->d_farray[vars.BX],
                                                                    fields_ptr->d_farray[vars.BY],
                                                                    fields_ptr->d_farray[vars.BZ]);
            }
        }
        if (param_ptr->boussinesq or param_ptr->heat_equation) {

            if(!param_ptr->spookyOutVar.name[i].compare(std::string("et"))) {
                // thermal/potential energy
                output_var = param_ptr->spookyOutVar.computeEnergy(fields_ptr->d_farray[vars.TH]);
            }
            else if(!param_ptr->spookyOutVar.name[i].compare(std::string("dissgradT"))) {
                // thermal dissipation (isotropic or anisotropic)
                // #if defined(ANISOTROPIC_DIFFUSION) && defined(MHD)
                if (param_ptr->anisotropic_diffusion and param_ptr->mhd) {
                // the minus sign is for consistency with snoopy
                    output_var = - param_ptr->spookyOutVar.computeAnisoDissipation(fields_ptr->d_all_fields,
                                                                            fields_ptr->d_all_buffer_r);
                }
                else {
                    output_var = param_ptr->spookyOutVar.computeDissipation(fields_ptr->d_farray[vars.TH]);
                }
            }
        }
        if (param_ptr->anisotropic_diffusion) {

            if(!param_ptr->spookyOutVar.name[i].compare(std::string("fluxbbgradT"))) {
                // thermal dissipation (isotropic or anisotropic)
                // for now it's only anisotropic
                output_var = param_ptr->spookyOutVar.computeAnisoInjection(fields_ptr->d_all_fields,
                                                                            fields_ptr->d_all_buffer_r);
            }
        }
        if (param_ptr->boussinesq) {
            if(!param_ptr->spookyOutVar.name[i].compare(std::string("thvx"))) {
                // convective flux
                output_var = param_ptr->spookyOutVar.twoFieldCorrelation(fields_ptr->d_farray_buffer_r[vars.TH], fields_ptr->d_farray_buffer_r[vars.VX]);
            }
            else if(!param_ptr->spookyOutVar.name[i].compare(std::string("thvz"))) {
                // convective flux
                output_var = param_ptr->spookyOutVar.twoFieldCorrelation(fields_ptr->d_farray_buffer_r[vars.TH], fields_ptr->d_farray_buffer_r[vars.VZ]);
            }
        }

        outputfile << std::scientific << std::setprecision(8) << output_var << "\t";
    }


    outputfile << "\n";
    outputfile.close();
}


void InputOutput::WriteTimevarOutputHeader() {

    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    if (param_ptr->debug > 0) {
        std::printf("Writing data output... \n");
    }

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


    outputfile << "\n";
    outputfile.close();
}
