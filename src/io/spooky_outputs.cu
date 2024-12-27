#include "common.hpp"
#include "fields.hpp"
// #include "parameters.hpp"
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

    data_type* curl = fields_ptr->d_all_tmparray;


    int blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
    Curl<<<blocksPerGrid, threadsPerBlock>>>(kvec, vx, curl, (size_t) grid.NTOTAL_COMPLEX);


    enstrophy = computeEnergy((data_type *)curl) + computeEnergy((data_type *)curl + grid.NTOTAL_COMPLEX) + computeEnergy((data_type *)curl + 2*grid.NTOTAL_COMPLEX) ;

    // if (stat != CUBLAS_STATUS_SUCCESS) std::printf("energy failed\n");

    return enstrophy;
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

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    if (param_ptr->anisotropic_diffusion) {

        scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;
        scalar_type* mask = fields_ptr->wavevector.d_mask;

        scalar_type* temperature = real_Buffer + 2 * grid.NTOTAL_COMPLEX * vars.TH;

        // this is the destination temp array for the divergence of the heat flux
        data_type* div_heat_flux = fields_ptr->d_tmparray[4];
        // set its elements to zero
        blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        VecInitComplex<<<blocksPerGrid, threadsPerBlock>>>(div_heat_flux, data_type(0.0,0.0), grid.NTOTAL_COMPLEX);

        supervisor_ptr->phys_ptr->AnisotropicConduction(complex_Fields, real_Buffer, div_heat_flux);

        // transform back to real
        c2r_fft(div_heat_flux, (scalar_type *) div_heat_flux);

        // compute 2field correlation between the divergence of bb grad T and T
        dissipation = twoFieldCorrelation( (scalar_type *) div_heat_flux, temperature);
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
    int blocksPerGrid;

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;
    scalar_type* mask = fields_ptr->wavevector.d_mask;

    if (param_ptr->anisotropic_diffusion and param_ptr->boussinesq) {

        scalar_type* bzb_vec = fields_ptr->d_tmparray_r[0];
        data_type* divbzb_vec = fields_ptr->d_tmparray[0];

        scalar_type* temperature = real_Buffer + 2 * grid.NTOTAL_COMPLEX * vars.TH;

        // Bx, By, Bz real fields are already in the 4-5-6 real_Buffer arrays
        scalar_type* mag_vec = real_Buffer + 2 * grid.NTOTAL_COMPLEX * vars.BX;
        // compute vector b_z \vec b (depending on which direction is the stratification)
        // and put it into the [num_fields - num_fields + 3] d_tmparray
        blocksPerGrid = ( 2 * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        Computebbstrat<<<blocksPerGrid, threadsPerBlock>>>( mag_vec, bzb_vec, (size_t) 2 * grid.NTOTAL_COMPLEX, STRAT_DIR);

        // transform to complex space
        for (int n = 0; n < 3; n++) {
            r2c_fft(bzb_vec + 2*n*grid.NTOTAL_COMPLEX, ((data_type*) bzb_vec) + n*grid.NTOTAL_COMPLEX);
        }

        // compute divergence of this vector
        blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        DivergenceMask<<<blocksPerGrid, threadsPerBlock>>>(kvec, (data_type*) bzb_vec, divbzb_vec, mask, grid.NTOTAL_COMPLEX, SET);

        // transform to real space
        c2r_fft(divbzb_vec, (scalar_type *) divbzb_vec);

        // compute 2 field correlation between div (b_z \vec b) and theta
        injection = twoFieldCorrelation( (scalar_type *) divbzb_vec, temperature);

    }

    return injection;

}
