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


    int blocksPerGrid;

    // we already computed the c2r FFTs when the WriteTimevarOutput()
    // was called, so the real fields are saved in the d_all_buffer_r
    scalar_type* real_velField = fields_ptr->d_all_buffer_r + vars.VEL * 2 * grid.NTOTAL_COMPLEX ;
    scalar_type* real_magField = fields_ptr->d_all_buffer_r + vars.MAG * 2 * grid.NTOTAL_COMPLEX ;
    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;
    scalar_type* mask = fields_ptr->wavevector.d_mask;

    /**
     * Now the nonlinear advection term (u \cdot \nabla) u
     *
     */

    if (param_ptr->incompressible) {

        // compute the elements of the shear symmetric matrix
        // S_ij = u_i u_j 
        // It has 6 independent components.
        // The results are saved in the temp_arrays from [0, 1, ..., 5]
        data_type* shear_matrix = fields_ptr->d_all_tmparray;
        data_type* divshear = fields_ptr->d_all_tmparray;

        blocksPerGrid = ( 2 * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        ShearMatrix<<<blocksPerGrid, threadsPerBlock>>>(real_velField, (scalar_type*) shear_matrix,  2 * grid.NTOTAL_COMPLEX);

        // take fft of 6 independent components of S_ij
        for (int n = 0; n < 6; n++) {
            r2c_fft((scalar_type*) shear_matrix + 2*n*grid.NTOTAL_COMPLEX, shear_matrix + n*grid.NTOTAL_COMPLEX, supervisor_ptr);
        }

        // compute derivative of shear matrix and assign it to divshear
        // (in fact we can reuse the first 3 temp_arrays)
        blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        NonLinAdvection<<<blocksPerGrid, threadsPerBlock>>>(kvec, shear_matrix, divshear, mask, grid.NTOTAL_COMPLEX);

        // now compute the nonlinear advection spectrum with the 3 components of divshear
        computeSpectrum1d(fields_ptr->d_farray[vars.VX],
                        divshear,
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "advec_x", output_spectrum, nbins);

        computeSpectrum1d(fields_ptr->d_farray[vars.VY],
                        divshear + grid.NTOTAL_COMPLEX,
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "advec_y", output_spectrum, nbins);

        computeSpectrum1d(fields_ptr->d_farray[vars.VZ],
                        divshear + 2*grid.NTOTAL_COMPLEX,
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "advec_z", output_spectrum, nbins);

    }

    /**
     * Then the emf spectra
     *
     */

    if (param_ptr->mhd) {

        // compute emf = u x B:
        // emf_x = u_y B_z - u_z B_y , emf_y = u_z B_x - u_x B_z , emf_z = u_x B_y - u_y B_x
        // the results are saved in the first 3 temp_arrays as [emf_x, emf_y, emf_z] (they are the x,y,z components of the emf)
        // We can re-utilize tmparrays and store result in in the temp_arrays from [0, 1, 2]

        data_type* emf = fields_ptr->d_all_tmparray;
        data_type* curlemf = fields_ptr->d_all_tmparray;

        blocksPerGrid = ( 2 * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        MagneticEmf<<<blocksPerGrid, threadsPerBlock>>>(real_velField, real_magField, (scalar_type*) emf,  2 * grid.NTOTAL_COMPLEX);

        // take fourier transforms of the 3 independent components of the antisymmetric shear matrix
        for (int n = 0; n < 3; n++) {
            r2c_fft((scalar_type*) emf + 2*n*grid.NTOTAL_COMPLEX, emf + n*grid.NTOTAL_COMPLEX, supervisor_ptr);
        }

        // compute curl of emf and assign to curlemf (we can re-utilize the temp arrays)
        blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        MagneticShear<<<blocksPerGrid, threadsPerBlock>>>(kvec, emf, curlemf, mask, grid.NTOTAL_COMPLEX);

        // now compute the spectrum for the 3 components
        computeSpectrum1d(fields_ptr->d_farray[vars.BX],
                        curlemf,
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "emfpower_x", output_spectrum, nbins);

        computeSpectrum1d(fields_ptr->d_farray[vars.BY],
                        curlemf + grid.NTOTAL_COMPLEX,
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "emfpower_y", output_spectrum, nbins);

        computeSpectrum1d(fields_ptr->d_farray[vars.BZ],
                        curlemf + 2*grid.NTOTAL_COMPLEX,
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "emfpower_z", output_spectrum, nbins);

    }

    /**
     * Then the Lorentz transfer spectra
     *
     */

    if (param_ptr->mhd) {

        // compute the elements of the shear symmetric matrix
        // S_ij = B_i B_j 
        // It has 6 independent components.
        // The results are saved in the temp_arrays from [0, 1, ..., 5]
        data_type* shear_matrix = fields_ptr->d_all_tmparray;
        data_type* divshear = fields_ptr->d_all_tmparray;

        blocksPerGrid = ( 2 * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        ShearMatrix<<<blocksPerGrid, threadsPerBlock>>>(real_magField, (scalar_type*) shear_matrix,  2 * grid.NTOTAL_COMPLEX);

        // take fft of 6 independent components of S_ij
        for (int n = 0; n < 6; n++) {
            r2c_fft((scalar_type*) shear_matrix + 2*n*grid.NTOTAL_COMPLEX, shear_matrix + n*grid.NTOTAL_COMPLEX, supervisor_ptr);
        }

        // compute derivative of shear matrix and assign it to divshear
        // (in fact we can reuse the first 3 temp_arrays)
        blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        NonLinAdvection<<<blocksPerGrid, threadsPerBlock>>>(kvec, shear_matrix, divshear, mask, grid.NTOTAL_COMPLEX);

        // now compute the Lorentz spectrum with the 3 components of divshear
        computeSpectrum1d(fields_ptr->d_farray[vars.VX],
                        divshear,
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "lorentz_x", output_spectrum, nbins);

        computeSpectrum1d(fields_ptr->d_farray[vars.VY],
                        divshear + grid.NTOTAL_COMPLEX,
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "lorentz_y", output_spectrum, nbins);

        computeSpectrum1d(fields_ptr->d_farray[vars.VZ],
                        divshear + 2*grid.NTOTAL_COMPLEX,
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "lorentz_z", output_spectrum, nbins);


    }

    /**
     * Then the nonlinear thermal advection 
     *
     */

    scalar_type* real_Theta = fields_ptr->d_all_buffer_r + vars.TH * 2 * grid.NTOTAL_COMPLEX ; 

    if (param_ptr->boussinesq) {

        data_type* en_flux = fields_ptr->d_all_tmparray;
        data_type* thermal_adv = fields_ptr->d_all_tmparray;

        // first compute energy flux vector [ u_x theta, u_y theta, u_z theta]
        // we can re-utilize tmparrays store result in in the temp_arrays from [0, 1, 2]
        blocksPerGrid = ( 2 * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        EnergyFluxVector<<<blocksPerGrid, threadsPerBlock>>>(real_velField, real_Theta, (scalar_type *) en_flux,  2 * grid.NTOTAL_COMPLEX);

        // take fourier transforms of the 3 energy flux vector components
        for (int n = 0; n < 3; n++) {
            r2c_fft(en_flux + 2*n*grid.NTOTAL_COMPLEX,  en_flux + n*grid.NTOTAL_COMPLEX, supervisor_ptr);
        }

        // compute derivative of energy flux vector and assign to thermal_adv
        // we can reuse the tmp_array
        blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        NonLinBoussinesqAdv<<<blocksPerGrid, threadsPerBlock>>>(kvec, en_flux, thermal_adv, mask, grid.NTOTAL_COMPLEX);

        // now compute the thermal nonlinear adv spectrum with thermal_adv
        computeSpectrum1d(fields_ptr->d_farray[vars.TH],
                        thermal_adv,
                        output_spectrum);
        writeSpectrumHelper(fname, time_save, "th_advec", output_spectrum, nbins);

    }

    if (param_ptr->anisotropic_diffusion) {

        // this is the destination temp array for the anisotropic
        // dissipation
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
