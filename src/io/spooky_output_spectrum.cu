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

void writeSpectrumHelper(std::string fname, double time_save, std::string name, double* output_spectrum, int nbins);

void InputOutput::WriteSpectrumOutput() {

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;
    std::shared_ptr<TimeStepping> timestep_ptr = supervisor_ptr->timestep_ptr;

    NVTX3_FUNC_RANGE();

    if (param_ptr->debug > 0) {
        std::printf("Writing spectrum output... \n");
    }


    double t0        = param_ptr->t_initial;
    double time_save = timestep_ptr->current_time;
    double tend     = param_ptr->t_final;
    double output_spectrum[nbins];

    char data_output_name[16];
    std::sprintf(data_output_name,"spectrum.spooky");
    std::string fname = param_ptr->output_dir + std::string("/data/") + std::string(data_output_name);


    /**
     * First the energies
     *
     */

    if (param_ptr->incompressible) {

        computeSpectrum1d(fields_ptr->d_farray[vars.VX],
                        fields_ptr->d_farray[vars.VX],
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "Kx", output_spectrum, nbins);

        computeSpectrum1d(fields_ptr->d_farray[vars.VY],
                        fields_ptr->d_farray[vars.VY],
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "Ky", output_spectrum, nbins);

        computeSpectrum1d(fields_ptr->d_farray[vars.VZ],
                        fields_ptr->d_farray[vars.VZ],
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "Kz", output_spectrum, nbins);

    }

    if (param_ptr->mhd) {

        computeSpectrum1d(fields_ptr->d_farray[vars.BX],
                        fields_ptr->d_farray[vars.BX],
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "Mx", output_spectrum, nbins);

        computeSpectrum1d(fields_ptr->d_farray[vars.BY],
                        fields_ptr->d_farray[vars.BY],
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "My", output_spectrum, nbins);

        computeSpectrum1d(fields_ptr->d_farray[vars.BZ],
                        fields_ptr->d_farray[vars.BZ],
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "Mz", output_spectrum, nbins);

    }

    if (param_ptr->boussinesq or param_ptr->heat_equation) {

        computeSpectrum1d(fields_ptr->d_farray[vars.TH],
                        fields_ptr->d_farray[vars.TH],
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "Eth", output_spectrum, nbins);

    }

    /**
     * Then the Reynolds/Maxwell/Buoyancy spectra
     *
     */

    if (param_ptr->incompressible) {

        computeSpectrum1d(fields_ptr->d_farray[vars.VX],
                        fields_ptr->d_farray[vars.VY],
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "vxvy", output_spectrum, nbins);

        computeSpectrum1d(fields_ptr->d_farray[vars.VY],
                        fields_ptr->d_farray[vars.VZ],
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "vyvz", output_spectrum, nbins);

        computeSpectrum1d(fields_ptr->d_farray[vars.VZ],
                        fields_ptr->d_farray[vars.VX],
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "vzvx", output_spectrum, nbins);

    }

    if (param_ptr->mhd) {

        computeSpectrum1d(fields_ptr->d_farray[vars.BX],
                        fields_ptr->d_farray[vars.BY],
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "bxby", output_spectrum, nbins);

        computeSpectrum1d(fields_ptr->d_farray[vars.BY],
                        fields_ptr->d_farray[vars.BZ],
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "bybz", output_spectrum, nbins);

        computeSpectrum1d(fields_ptr->d_farray[vars.BZ],
                        fields_ptr->d_farray[vars.BX],
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "bzbx", output_spectrum, nbins);

    }

    if (param_ptr->boussinesq) {

        computeSpectrum1d(fields_ptr->d_farray[vars.TH],
                        fields_ptr->d_farray[vars.VX],
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "thvx", output_spectrum, nbins);

        computeSpectrum1d(fields_ptr->d_farray[vars.TH],
                        fields_ptr->d_farray[vars.VY],
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "thvy", output_spectrum, nbins);

        computeSpectrum1d(fields_ptr->d_farray[vars.TH],
                        fields_ptr->d_farray[vars.VZ],
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "thvz", output_spectrum, nbins);

    }



    

}


void writeSpectrumHelper(std::string fname, double time_save, std::string name, double* output_spectrum, int nbins){

    std::ofstream outputfile;
    outputfile.open (fname, std::ios_base::app);

    outputfile << "t" << "\t";
    outputfile << std::scientific << std::setprecision(8) << time_save << "\t";

    outputfile << name << "\t";

    for (int i = 0; i < nbins; i++) {
        outputfile << std::scientific << std::setprecision(8) << output_spectrum[i] << "\t";
    }

    outputfile << "\n";

    outputfile.close();


}

void InputOutput::computeSpectrum1d(data_type* v1, data_type* v2,
                       double* output_spectrum) {

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;
    int nbins = fields_ptr->wavevector.nbins;
    double deltak = fields_ptr->wavevector.deltak;

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;

    int blocksPerGrid = ( nbins + threadsPerBlock - 1) / threadsPerBlock;
    VecInit<<<blocksPerGrid, threadsPerBlock>>>(d_output_spectrum, 0.0, nbins);


    blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
    Spectrum1d<<<blocksPerGrid, threadsPerBlock>>>(kvec, v1, v2, d_output_spectrum, nbins, deltak, grid.NX, grid.NY, grid.NZ, (size_t) grid.NTOTAL_COMPLEX);

    CUDA_RT_CALL(cudaMemcpy(output_spectrum, d_output_spectrum, sizeof(scalar_type) * nbins, cudaMemcpyDeviceToHost));


}



void InputOutput::WriteSpectrumOutputHeader() {

    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    int nbins = supervisor_ptr->fields_ptr->wavevector.nbins;
    double deltak = supervisor_ptr->fields_ptr->wavevector.deltak;

    if (param_ptr->debug > 0) {
        std::printf("Writing spectrum header... \n");
    }

    char data_output_name[16];
    std::sprintf(data_output_name,"spectrum.spooky");
    std::string fname = param_ptr->output_dir + std::string("/data/") + std::string(data_output_name);

    std::ofstream outputfile;
    outputfile.open (fname, std::ios_base::app);


    outputfile << "## This file contains the 1d (shell-integrated) energy spectral densities of the following quantities: \n";
    outputfile << "## \t";

    // for(int i = 0 ; i < spookyOutSpectrum.size() ; i++) {
    //     outputfile << spookyOutSpectrum[i]  << "\t";
    // }

    outputfile << "## The wavevector: \n";

    for(int i = 0 ; i < nbins ; i++) {
        outputfile << std::scientific << std::setprecision(8) << i*deltak << "\t";
    }


    outputfile << "\n";
    outputfile.close();
}
