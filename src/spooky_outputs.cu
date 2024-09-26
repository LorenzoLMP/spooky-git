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

SpookyOutput::SpookyOutput() {
    // double lx, ly, lz;
    // read_Parameters();
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

scalar_type SpookyOutput::computeEnstrophy(data_type *v_all_complex,
                                             scalar_type *d_all_kvec,
                                             data_type *tmparray) {
    /***
     * This function uses complex inputs to compute the "enstrophy" of a vector field
     * To do so, we first compute the curl of the field, and then sum the "energies" of the
     * three components. We divide by (1/2) in agreement with the definition of the enstrophy.
     ***/

    // cublasStatus_t stat;
    // scalar_type norm = 0.0;
    scalar_type enstrophy = 0.0;


    int blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    Curl<<<blocksPerGrid, threadsPerBlock>>>(d_all_kvec, v_all_complex, tmparray, (size_t) ntotal_complex);


    enstrophy = computeEnergy((data_type *)tmparray) + computeEnergy((data_type *)tmparray + ntotal_complex) + computeEnergy((data_type *)tmparray + 2*ntotal_complex) ;

    // if (stat != CUBLAS_STATUS_SUCCESS) std::printf("energy failed\n");

    return 0.5*enstrophy;
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

