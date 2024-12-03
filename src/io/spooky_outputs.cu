// #include "define_types.hpp"
// // #include "fields.hpp"
// #include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
// #include "fields.hpp"
// #include "parameters.hpp"
#include "spooky_outputs.hpp"
#include "cublas_routines.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "cufft_routines.hpp"
#include "user_defined_cuda_kernels.hpp"
#include "supervisor.hpp"

SpookyOutput::SpookyOutput() {
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


    stat = cublasDznrm2(handle0, ntotal_complex, (cuDoubleComplex *) vcomplex, 1, &tmp);


    energy += tmp*tmp/(ntotal*ntotal);

    // ok
    stat = cublasDznrm2(handle0, nx*ny, (cuDoubleComplex *)vcomplex, (nz/2+1), &subtract);

    energy -= 0.5*subtract*subtract/(ntotal*ntotal);

    // this sums all k=1 modes for each i,j
    // stat = cublasDznrm2(handle0, nx*ny, (cuDoubleComplex *)v1complex + 1, (nz/2+1), &subtract);


    if (stat != CUBLAS_STATUS_SUCCESS) std::printf("energy failed\n");

    return energy;
}

scalar_type SpookyOutput::computeEnstrophy(data_type *vx,
                                           data_type *vy,
                                           data_type *vz) {
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

    double_type* curl = fields_ptr->d_all_tmparray;


    int blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    Curl<<<blocksPerGrid, threadsPerBlock>>>(kvec, vx, curl, (size_t) ntotal_complex);


    enstrophy = computeEnergy((data_type *)curl) + computeEnergy((data_type *)curl + ntotal_complex) + computeEnergy((data_type *)curl + 2*ntotal_complex) ;

    // if (stat != CUBLAS_STATUS_SUCCESS) std::printf("energy failed\n");

    return enstrophy;
}

scalar_type SpookyOutput::computeDissipation(data_type *scalar_complex,
                                             scalar_type *d_all_kvec,
                                             data_type *tmparray) {
    /***
     * This function uses complex inputs to compute the "dissipation" of a scalar field (-k^2 th^2)
     * To do so, we first compute the gradient of the field, and then sum the "energies" of the
     * three components.
     ***/

    // cublasStatus_t stat;
    // scalar_type norm = 0.0;
    scalar_type dissipation = 0.0;


    int blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    Gradient<<<blocksPerGrid, threadsPerBlock>>>(d_all_kvec, scalar_complex, tmparray, (size_t) ntotal_complex);


    dissipation = computeEnergy((data_type *)tmparray) + computeEnergy((data_type *)tmparray + ntotal_complex) + computeEnergy((data_type *)tmparray + 2*ntotal_complex) ;

    // if (stat != CUBLAS_STATUS_SUCCESS) std::printf("energy failed\n");

    return dissipation;
}



scalar_type SpookyOutput::twoFieldCorrelation( scalar_type *v1,
                                               scalar_type *v2) {
    /***
     * This function uses real inputs to compute the correlation between
     * 2 fields.
     * Because of the way the 3d array is set up with rFFTs,
     * the real field has dimensions: nx, ny, nz+2.
     * The last two "rows" k = nz and k = nz + 1 (k = 0, 1, ...)
     * are not guaranteed to be always zero. For this reason,
     * we first run the sum over the entire array, k = nz, nz+1
     * included, and then we subtract the two rows.
     ***/

    cublasStatus_t stat;
    // scalar_type norm = 0.0;
    scalar_type correlation = 0.0;
    scalar_type subtract = 0.0;
    scalar_type tmp = 0.0;

    // stat = cublasDznrm2(handle0, ntotal_complex, (cuDoubleComplex *) vcomplex, 1, &tmp);
    stat = cublasDdot(handle0, 2*ntotal_complex,
                           v1, 1, v2, 1, &tmp);


    correlation += tmp/(ntotal);

    // subtract k = nz terms
    stat = cublasDdot(handle0, nx*ny,
                        v1 + nz, nz + 2,
                        v2 + nz, nz + 2, &subtract);

    correlation -= subtract/(ntotal);

    // subtract k = nz + 1 terms
    stat = cublasDdot(handle0, nx*ny,
                        v1 + nz + 1, nz + 2,
                        v2 + nz + 1, nz + 2, &subtract);

    correlation -= subtract/(ntotal);

    // this sums all k=1 modes for each i,j
    // stat = cublasDznrm2(handle0, nx*ny, (cuDoubleComplex *)v1complex + 1, (nz/2+1), &subtract);


    if (stat != CUBLAS_STATUS_SUCCESS) std::printf("2corr failed\n");

    return correlation;
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
    int blocksPerGrid;

#ifdef BOUSSINESQ
#ifdef MHD
#ifdef ANISOTROPIC_DIFFUSION

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;
    scalar_type* mask = fields_ptr->wavevector.d_mask;

    scalar_type* temperature = real_Buffer + 2 * ntotal_complex * TH;

    // this is the destination temp array for the divergence of the heat flux
    data_type* div_heat_flux = fields_ptr->d_tmparray[4];
    // set its elements to zero
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock
    VecInitComplex<<<blocksPerGrid, threadsPerBlock>>>(div_heat_flux, data_type(0.0,0.0), ntotal_complex);

    AnisotropicConduction(complex_Fields, real_Buffer, div_heat_flux);

    // transform back to real
    c2r_fft(div_heat_flux, (scalar_type *) div_heat_flux);

    // compute 2field correlation between the divergence of bb grad T and T
    dissipation = twoFieldCorrelation( (scalar_type *) div_heat_flux, temperature);


#endif // ANISOTROPIC_DIFFUSION
#endif // MHD
#endif // BOUSSINESQ

    return dissipation;

}

scalar_type SpookyOutput::computeAnisoInjection(data_type* complex_Fields, scalar_type* real_Buffer) {
    /***
     * This function uses complex inputs to compute the anisotropic "injection" with the MTI
     * To do so, we compute the MTI injecttion term div (b b_z) transform to real,
     * then compute the 2fieldcorrelation between this term and theta
     ***/

    scalar_type injection = 0.0;
    int blocksPerGrid;

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;
    scalar_type* mask = fields_ptr->wavevector.d_mask;

#ifdef BOUSSINESQ
#ifdef MHD
#ifdef ANISOTROPIC_DIFFUSION


    scalar_type* bzb_vec = fields_ptr->d_tmparray_r[0];
    data_type* divbzb_vec = fields_ptr->d_tmparray[0];

    scalar_type* temperature = real_Buffer + 2 * ntotal_complex * TH;

    // Bx, By, Bz real fields are already in the 4-5-6 real_Buffer arrays
    scalar_type* mag_vec = real_Buffer + 2 * ntotal_complex * BX;
    // compute vector b_z \vec b (depending on which direction is the stratification)
    // and put it into the [num_fields - num_fields + 3] d_tmparray
    blocksPerGrid = ( 2 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    Computebbstrat<<<blocksPerGrid, threadsPerBlock>>>( mag_vec, bzb_vec, (size_t) 2 * ntotal_complex, STRAT_DIR);

    // transform to complex space
    for (int n = 0; n < 3; n++) {
        r2c_fft(bzb_vec + 2*n*ntotal_complex, ((data_type*) bzb_vec) + n*ntotal_complex);
    }

    // compute divergence of this vector
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    DivergenceMask<<<blocksPerGrid, threadsPerBlock>>>(kvec, (data_type*) bzb_vec, divbzb_vec, mask, ntotal_complex, ASS);

    // transform to real space
    c2r_fft(divbzb_vec, (scalar_type *) divbzb_vec);

    // compute 2 field correlation between div (b_z \vec b) and theta
    injection= twoFieldCorrelation( divbzb_vec, temperature);


#endif // ANISOTROPIC_DIFFUSION
#endif // MHD
#endif // BOUSSINESQ

    return injection;

}
