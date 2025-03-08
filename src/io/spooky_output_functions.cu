#include "common.hpp"
#include "fields.hpp"
#include "parameters.hpp"
#include "spooky_outputs.hpp"
#include "cublas_routines.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "cufft_routines.hpp"
#include "user_defined_cuda_kernels.hpp"
#include "supervisor.hpp"
#include "physics.hpp"

SpookyOutput::SpookyOutput(Supervisor &sup_in) {
    // double lx, ly, lz;
    // read_Parameters();
    supervisor_ptr = &sup_in;
}

SpookyOutput::~SpookyOutput() {
}


scalar_type SpookyOutput::computeEnergy( data_type *vcomplex ) {
    /***
     * This function uses complex input to compute the "energy"
     * The modes with k>0 only have half the energy (because the k<0 is not present).
     * Here we multiply all k modes by 2 and then subtract once the energy in the k=0 mode.
     * The total is then divided by 2 to give quantity (i.e. Energy ~ (1/2) v^2)
     ***/

    cublasStatus_t stat;
    // scalar_type norm = 0.0;
    scalar_type energy = 0.0;
    scalar_type subtract = 0.0;
    scalar_type tmp = 0.0;


    stat = cublasDznrm2(handle0, grid.NTOTAL_COMPLEX, (cuDoubleComplex *) vcomplex, 1, &tmp);


    energy += tmp*tmp/(grid.NTOTAL*grid.NTOTAL);

    // ok
    stat = cublasDznrm2(handle0, grid.NX*grid.NY, (cuDoubleComplex *)vcomplex, (grid.NZ/2+1), &subtract);

    energy -= 0.5*subtract*subtract/(grid.NTOTAL*grid.NTOTAL);

    // this sums all k=1 modes for each i,j
    // stat = cublasDznrm2(handle0, grid.NX*grid.NY, (cuDoubleComplex *)v1complex + 1, (grid.NZ/2+1), &subtract);


    if (stat != CUBLAS_STATUS_SUCCESS) std::printf("energy failed\n");

    return energy;
}

scalar_type SpookyOutput::computeEnstrophy(data_type *complex_vecField) {
    /***
     * This function uses complex inputs to compute the "enstrophy" of a vector field
     * To do so, we first compute the curl of the field, and then sum the "energies" of the
     * three components.
     ***/

    // cublasStatus_t stat;
    // scalar_type norm = 0.0;
    scalar_type enstrophy = 0.0;

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    // std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;
    // scalar_type* mask = fields_ptr->wavevector.d_mask;

    data_type* curl = fields_ptr->d_all_tmparray;


    int blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
    Curl<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_vecField, curl, (size_t) grid.NTOTAL_COMPLEX);


    enstrophy = computeEnergy((data_type *)curl) + computeEnergy((data_type *)curl + grid.NTOTAL_COMPLEX) + computeEnergy((data_type *)curl + 2*grid.NTOTAL_COMPLEX) ;

    // if (stat != CUBLAS_STATUS_SUCCESS) std::printf("energy failed\n");

    return enstrophy;
}

scalar_type SpookyOutput::computeHelicity(data_type *complex_magField, scalar_type *real_magField) {
    /***
     * This function uses complex inputs to compute the magnetic helicity field
     * Can also work for kinetic helicity.
     * 
     ***/

    // cublasStatus_t stat;
    // scalar_type norm = 0.0;
    scalar_type helicity = 0.0;

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    // std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;
    // scalar_type* mask = fields_ptr->wavevector.d_mask;

    data_type* mag_helicity = fields_ptr->d_all_tmparray;

    supervisor_ptr->phys_ptr->MagneticHelicity(complex_magField, mag_helicity);

    // transform back to real
    for (int n = 0; n < 3; n++) {
        c2r_fft(mag_helicity + n*grid.NTOTAL_COMPLEX, (scalar_type *) mag_helicity + 2*n*grid.NTOTAL_COMPLEX);
    }

    // compute scalar product  
    for (int n = 0; n < 3; n++) {
        helicity += twoFieldCorrelation( (scalar_type *) mag_helicity + 2*n*grid.NTOTAL_COMPLEX, real_magField + 2*n*grid.NTOTAL_COMPLEX);
    }

    return helicity;
}

scalar_type SpookyOutput::potentialVorticity(data_type *complex_velField, data_type *complex_Theta) {
    /***
     * This function uses complex inputs to compute the potential vorticity
     * 
     ***/

    scalar_type pv = 0.0;

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    // std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;
    // scalar_type* mask = fields_ptr->wavevector.d_mask;

    data_type* curl_vel = fields_ptr->d_all_tmparray;

    int blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
    Curl<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_velField, curl_vel, grid.NTOTAL_COMPLEX);

    data_type* grad_theta = fields_ptr->d_tmparray[3];
    Gradient<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_Theta, grad_theta, grid.NTOTAL_COMPLEX);


    // transform back to real
    for (int n = 0; n < 3; n++) {
        c2r_fft(curl_vel + n*grid.NTOTAL_COMPLEX, (scalar_type *) curl_vel + 2*n*grid.NTOTAL_COMPLEX);
        c2r_fft(grad_theta + n*grid.NTOTAL_COMPLEX, (scalar_type *) grad_theta + 2*n*grid.NTOTAL_COMPLEX);
    }

    // compute scalar product  
    for (int n = 0; n < 3; n++) {
        pv += twoFieldCorrelation( (scalar_type *) curl_vel + 2*n*grid.NTOTAL_COMPLEX, (scalar_type *) grad_theta  + 2*n*grid.NTOTAL_COMPLEX);
    }

    return pv;
}

scalar_type SpookyOutput::computeDissipation(data_type *scalar_complex) {
    /***
     * This function uses complex inputs to compute the "dissipation" of a scalar field (-k^2 th^2)
     * To do so, we first compute the gradient of the field, and then sum the "energies" of the
     * three components.
     ***/

    // cublasStatus_t stat;
    // scalar_type norm = 0.0;
    scalar_type dissipation = 0.0;
    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;

    // tmp array
    data_type* tmparray = fields_ptr->d_tmparray[0];
    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;

    int blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
    Gradient<<<blocksPerGrid, threadsPerBlock>>>(kvec, scalar_complex, tmparray, (size_t) grid.NTOTAL_COMPLEX);


    dissipation = computeEnergy((data_type *)tmparray) + computeEnergy((data_type *)tmparray + grid.NTOTAL_COMPLEX) + computeEnergy((data_type *)tmparray + 2*grid.NTOTAL_COMPLEX) ;

    // if (stat != CUBLAS_STATUS_SUCCESS) std::printf("energy failed\n");

    return dissipation;
}



scalar_type SpookyOutput::twoFieldCorrelation( scalar_type *v1,
                                               scalar_type *v2) {
    /***
     * This function uses real inputs to compute the correlation between
     * 2 fields.
     * Because of the way the 3d array is set up with rFFTs,
     * the real field has dimensions: grid.NX, grid.NY, grid.NZ+2.
     * The last two "rows" k = grid.NZ and k = grid.NZ + 1 (k = 0, 1, ...)
     * are not guaranteed to be always zero. For this reason,
     * we first run the sum over the entire array, k = grid.NZ, grid.NZ+1
     * included, and then we subtract the two rows.
     ***/

    cublasStatus_t stat;
    // scalar_type norm = 0.0;
    scalar_type correlation = 0.0;
    scalar_type subtract = 0.0;
    scalar_type tmp = 0.0;

    // stat = cublasDznrm2(handle0, grid.NTOTAL_COMPLEX, (cuDoubleComplex *) vcomplex, 1, &tmp);
    stat = cublasDdot(handle0, 2*grid.NTOTAL_COMPLEX,
                           v1, 1, v2, 1, &tmp);


    correlation += tmp/(grid.NTOTAL);

    // subtract k = grid.NZ terms
    stat = cublasDdot(handle0, grid.NX*grid.NY,
                        v1 + grid.NZ, grid.NZ + 2,
                        v2 + grid.NZ, grid.NZ + 2, &subtract);

    correlation -= subtract/(grid.NTOTAL);

    // subtract k = grid.NZ + 1 terms
    stat = cublasDdot(handle0, grid.NX*grid.NY,
                        v1 + grid.NZ + 1, grid.NZ + 2,
                        v2 + grid.NZ + 1, grid.NZ + 2, &subtract);

    correlation -= subtract/(grid.NTOTAL);

    // this sums all k=1 modes for each i,j
    // stat = cublasDznrm2(handle0, grid.NX*grid.NY, (cuDoubleComplex *)v1complex + 1, (grid.NZ/2+1), &subtract);


    if (stat != CUBLAS_STATUS_SUCCESS) std::printf("2corr failed\n");

    return correlation;
}


scalar_type SpookyOutput::oneFieldAverage( scalar_type *v1) {
    // we use the following trick: we re-utilize the twoFieldCorrelation
    // function with one of the inputs being made of ones.
    // to avoid overwriting the input we choose the second unit vector
    // to be next in memory. Only thing is to make sure we are not going
    // beyond the allocated temp arrays (6 of them)

    scalar_type average = 0.0;
    scalar_type* unit_vec = v1 + 2 * grid.NTOTAL_COMPLEX ;

    int blocksPerGrid = ( 2 * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
    VecInit<<<blocksPerGrid, threadsPerBlock>>>(unit_vec, 1.0, 2 * grid.NTOTAL_COMPLEX);

    average = twoFieldCorrelation( v1, unit_vec);

    return average;

}

scalar_type SpookyOutput::computeAnisoDissipation(data_type* complex_Fields, scalar_type* real_Buffer) {
    /***
     * This function uses complex inputs to compute the anisotropic "dissipation" with
     * anisotropic thermal conduction along magnetic field lines.
     * To do so, we compute the term div (\vec b b \cdot \grad theta ) transform to real,
     * then compute the 2fieldcorrelation between this term and theta
     ***/

    // cublasStatus_t stat;
    // scalar_type norm = 0.0;
    scalar_type dissipation = 0.0;
    // int blocksPerGrid;

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    if (param_ptr->anisotropic_diffusion) {

        // scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;
        // scalar_type* mask = fields_ptr->wavevector.d_mask;

        scalar_type* temperature = real_Buffer + 2 * grid.NTOTAL_COMPLEX * vars.TH;

        // this is the destination temp array for the divergence of the heat flux
        data_type* aniso_dissipation = fields_ptr->d_tmparray[4];
        // set its elements to zero
        // blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        // VecInitComplex<<<blocksPerGrid, threadsPerBlock>>>(div_heat_flux, data_type(0.0,0.0), grid.NTOTAL_COMPLEX);

        // WARNING: I think this does the entire anisotropic heat flux
        // including dissipation and injection
        // Now it should just do the dissipation (but needs to be checked)
        supervisor_ptr->phys_ptr->AnisotropicDissipation(complex_Fields, real_Buffer, aniso_dissipation);

        // transform back to real
        c2r_fft(aniso_dissipation, (scalar_type *) aniso_dissipation);

        // compute 2field correlation between the divergence of bb grad T and T
        dissipation = twoFieldCorrelation( (scalar_type *) aniso_dissipation, temperature);
    }

    return dissipation;

}

scalar_type SpookyOutput::computeAnisoInjection(data_type* complex_Fields, scalar_type* real_Buffer) {
    /***
     * This function uses complex inputs to compute the anisotropic "injection" with the MTI
     * To do so, we compute the MTI injecttion term div (b b_z) transform to real,
     * then compute the 2fieldcorrelation between this term and theta
     ***/

    scalar_type injection = 0.0;
    // int blocksPerGrid;

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;


    if (param_ptr->anisotropic_diffusion and param_ptr->boussinesq) {

        data_type* anisoInjVec = fields_ptr->d_tmparray[0];

        supervisor_ptr->phys_ptr->AnisotropicInjection(complex_Fields, real_Buffer, anisoInjVec);

    //     scalar_type* bzb_vec = fields_ptr->d_tmparray_r[0];

        scalar_type* temperature = real_Buffer + 2 * grid.NTOTAL_COMPLEX * vars.TH;

        // transform to real space
        c2r_fft(anisoInjVec, (scalar_type *) anisoInjVec);

        // compute 2 field correlation between div (b_z \vec b) and theta
        injection = twoFieldCorrelation( (scalar_type *) anisoInjVec, temperature);

    }

    return injection;

}

scalar_type SpookyOutput::averagebz(scalar_type* real_magField) {

    scalar_type average_bz = 0.0;
    // int blocksPerGrid;

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    // std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    scalar_type* bvec = fields_ptr->d_tmparray_r[0];
    scalar_type* bz = fields_ptr->d_tmparray_r[2];

    int blocksPerGrid = ( 2 * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
    bUnitvector<<<blocksPerGrid, threadsPerBlock>>>(real_magField, bvec, 2 * grid.NTOTAL_COMPLEX);

    // we can overwrite bz
    scalar_type* abs_bz = fields_ptr->d_tmparray_r[2];
    DoubleAbsolute<<<blocksPerGrid, threadsPerBlock>>>(bz , abs_bz,  2 * grid.NTOTAL_COMPLEX);

    average_bz = oneFieldAverage(abs_bz);

    return average_bz;
}


scalar_type SpookyOutput::averagebz2(scalar_type* real_magField) {

    scalar_type average_bz2 = 0.0;
    // int blocksPerGrid;

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    // std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    scalar_type* bvec = fields_ptr->d_tmparray_r[0];
    scalar_type* bz = fields_ptr->d_tmparray_r[2];

    int blocksPerGrid = ( 2 * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
    bUnitvector<<<blocksPerGrid, threadsPerBlock>>>(real_magField, bvec, 2 * grid.NTOTAL_COMPLEX);

    average_bz2 = twoFieldCorrelation(bz,bz);

    return average_bz2;
}

scalar_type SpookyOutput::averagephiB(scalar_type* real_magField) {

    scalar_type average_phiB = 0.0;
    // int blocksPerGrid;

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    // std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    scalar_type* phi = fields_ptr->d_tmparray_r[0];


    int blocksPerGrid = ( 2 * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
    AngleHorizPlane<<<blocksPerGrid, threadsPerBlock>>>(real_magField, phi, 2 * grid.NTOTAL_COMPLEX);

    average_phiB = oneFieldAverage(phi);

    return average_phiB;
}