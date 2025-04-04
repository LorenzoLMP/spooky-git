#include "common.hpp"
#include "cufft_routines.hpp"
#include "cuda_kernels_generic.hpp"
#include "cuda_kernels.hpp"
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
#include "physics.hpp"

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

    for (int i = 0; i < param_ptr->spookyOutVar.length_spectra; i++){

        // reset the output spectrum
        for (int n = 0; n < nbins; n++) {
            output_spectrum[n] = -1.0;
        }
        /**
        * First the energies
        *
        */

        if (param_ptr->incompressible) {
            
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("Kx"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.VX],
                                fields_ptr->d_farray[vars.VX],
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "Kx", output_spectrum, nbins);
            }
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("Ky"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.VY],
                                fields_ptr->d_farray[vars.VY],
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "Ky", output_spectrum, nbins);
            }
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("Kz"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.VZ],
                                fields_ptr->d_farray[vars.VZ],
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "Kz", output_spectrum, nbins);
            }

        }

        if (param_ptr->mhd) {

            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("Mx"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.BX],
                                fields_ptr->d_farray[vars.BX],
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "Mx", output_spectrum, nbins);
            }
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("My"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.BY],
                                fields_ptr->d_farray[vars.BY],
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "My", output_spectrum, nbins);
            }
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("Mz"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.BZ],
                                fields_ptr->d_farray[vars.BZ],
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "Mz", output_spectrum, nbins);
            }

        }

        if (param_ptr->boussinesq or param_ptr->heat_equation) {

            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("Eth"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.TH],
                                fields_ptr->d_farray[vars.TH],
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "Eth", output_spectrum, nbins);
            }

        }

        /**
        * Then the Reynolds/Maxwell/Buoyancy spectra
        *
        */

        if (param_ptr->incompressible) {

            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("vxvy"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.VX],
                                fields_ptr->d_farray[vars.VY],
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "vxvy", output_spectrum, nbins);
            }
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("vyvz"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.VY],
                                fields_ptr->d_farray[vars.VZ],
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "vyvz", output_spectrum, nbins);
            }
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("vxvz"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.VX],
                                fields_ptr->d_farray[vars.VZ],
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "vxvz", output_spectrum, nbins);
            }

        }

        if (param_ptr->mhd) {

            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("bxby"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.BX],
                                fields_ptr->d_farray[vars.BY],
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "bxby", output_spectrum, nbins);
            }
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("bybz"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.BY],
                                fields_ptr->d_farray[vars.BZ],
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "bybz", output_spectrum, nbins);
            }
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("bxbz"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.BX],
                                fields_ptr->d_farray[vars.BY],
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "bxbz", output_spectrum, nbins);
            }

        }

        if (param_ptr->boussinesq) {

            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("thvx"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.TH],
                                fields_ptr->d_farray[vars.VX],
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "thvx", output_spectrum, nbins);
            }
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("thvy"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.TH],
                                fields_ptr->d_farray[vars.VY],
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "thvy", output_spectrum, nbins);
            }
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("thvz"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.TH],
                                fields_ptr->d_farray[vars.VZ],
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "thvz", output_spectrum, nbins);
            }

        }


        // // we already computed the c2r FFTs when the WriteTimevarOutput()
        // // was called, so the real fields are saved in the d_all_buffer_r
        
        /**
        * Now the nonlinear advection term (u \cdot \nabla) u
        *
        */

        if (param_ptr->incompressible) {

            scalar_type* real_velField = fields_ptr->d_all_buffer_r + vars.VEL * 2 * grid.NTOTAL_COMPLEX ;

            data_type* divshear = fields_ptr->d_all_tmparray;

            supervisor_ptr->phys_ptr->NonLinearAdvection(real_velField, divshear);
            
            // now compute the nonlinear advection spectrum with the 3 components of divshear
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("advec_x"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.VX],
                                divshear,
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "advec_x", output_spectrum, nbins);
            }
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("advec_y"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.VY],
                                divshear + grid.NTOTAL_COMPLEX,
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "advec_y", output_spectrum, nbins);
            }
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("advec_z"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.VZ],
                                divshear + 2*grid.NTOTAL_COMPLEX,
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "advec_z", output_spectrum, nbins);
            }

        }

        /**
        * Then the emf spectra
        *
        */

        if (param_ptr->mhd) {

            data_type* curlemf = fields_ptr->d_all_tmparray;
            supervisor_ptr->phys_ptr->CurlEMF(fields_ptr->d_all_fields, fields_ptr->d_all_buffer_r, curlemf);


            // now compute the spectrum for the 3 components
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("emfpower_x"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.BX],
                                curlemf,
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "emfpower_x", output_spectrum, nbins);
            }
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("emfpower_y"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.BY],
                                curlemf + grid.NTOTAL_COMPLEX,
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "emfpower_y", output_spectrum, nbins);
            }
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("emfpower_z"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.BZ],
                                curlemf + 2*grid.NTOTAL_COMPLEX,
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "emfpower_z", output_spectrum, nbins);
            }

        }

        /**
        * Then the Lorentz transfer spectra
        *
        */

        if (param_ptr->mhd) {

            scalar_type* real_magField = fields_ptr->d_all_buffer_r + vars.MAG * 2 * grid.NTOTAL_COMPLEX ;

            data_type* divshear = fields_ptr->d_all_tmparray;

            supervisor_ptr->phys_ptr->NonLinearAdvection(real_magField, divshear);

            // now compute the Lorentz spectrum with the 3 components of divshear
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("lorentz_x"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.VX],
                                divshear,
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "lorentz_x", output_spectrum, nbins);
            }
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("lorentz_y"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.VY],
                                divshear + grid.NTOTAL_COMPLEX,
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "lorentz_y", output_spectrum, nbins);
            }
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("lorentz_z"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.VZ],
                                divshear + 2*grid.NTOTAL_COMPLEX,
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "lorentz_z", output_spectrum, nbins);
            }


        }

        /**
        * Then the nonlinear thermal advection 
        *
        */

        if (param_ptr->boussinesq) {

            data_type* thermal_adv = fields_ptr->d_all_tmparray;

            supervisor_ptr->phys_ptr->AdvectTemperature(fields_ptr->d_all_fields, fields_ptr->d_all_buffer_r, thermal_adv);

            // now compute the thermal nonlinear adv spectrum with thermal_adv
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("th_advec"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.TH],
                                thermal_adv,
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "th_advec", output_spectrum, nbins);
            }

        }


        /**
        * These are the helicity spectra
        * 
        */

        if (param_ptr->mhd) {

            data_type* complex_dMag = fields_ptr->d_all_fields + vars.MAG * grid.NTOTAL_COMPLEX ;
            data_type* mag_helicity = fields_ptr->d_all_tmparray;

            supervisor_ptr->phys_ptr->MagneticHelicity(complex_dMag, mag_helicity);

            // now compute the Helicity spectra with the 3 components of mag_helicity
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("helicity_x"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.BX],
                                mag_helicity,
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "helicity_x", output_spectrum, nbins);
            }
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("helicity_y"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.BY],
                                mag_helicity + grid.NTOTAL_COMPLEX,
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "helicity_y", output_spectrum, nbins);
            }
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("helicity_z"))) {
                computeSpectrum1d(fields_ptr->d_farray[vars.BZ],
                                mag_helicity + 2*grid.NTOTAL_COMPLEX,
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "helicity_z", output_spectrum, nbins);
            }
            
        }



        /**
        * These are the anisotropic conduction terms
        * 
        */

        // first the dissipation
        if (param_ptr->anisotropic_diffusion) {

            // this is the destination temp array for the anisotropic
            // dissipation
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("th_aniso_diss"))) {
                data_type* aniso_dissipation = fields_ptr->d_tmparray[4];
                
                supervisor_ptr->phys_ptr->AnisotropicDissipation(fields_ptr->d_all_fields, 
                                                                fields_ptr->d_all_buffer_r, 
                                                                aniso_dissipation);
                
                computeSpectrum1d(fields_ptr->d_farray[vars.TH],
                                aniso_dissipation,
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "th_aniso_diss", output_spectrum, nbins);
            }

        }

        // then the anisotropic injection
        if (param_ptr->anisotropic_diffusion) {

            // this is the destination temp array for the anisotropic
            // injection
            if(!param_ptr->spookyOutVar.name_spectra[i].compare(std::string("th_aniso_inj"))) {
                data_type* aniso_injection = fields_ptr->d_tmparray[4];
                
                supervisor_ptr->phys_ptr->AnisotropicInjection(fields_ptr->d_all_fields, 
                                                                fields_ptr->d_all_buffer_r, 
                                                                aniso_injection);
                
                computeSpectrum1d(fields_ptr->d_farray[vars.TH],
                                aniso_injection,
                                output_spectrum);
                writeSpectrumHelper(fname, time_save, "th_aniso_inj", output_spectrum, nbins);
            }

        }

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
    for(int i = 0 ; i < param_ptr->spookyOutVar.length_spectra ; i++) {
        outputfile << param_ptr->spookyOutVar.name_spectra[i]  << "\t";
    }
    outputfile << "\n";


    outputfile << "wavevector \t";
    for(int i = 0 ; i < nbins ; i++) {
        outputfile << std::scientific << std::setprecision(8) << i*deltak << "\t";
    }
    outputfile << "\n";

    outputfile.close();
}
