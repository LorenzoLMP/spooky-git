#include "define_types.hpp"
#include "fields.hpp"
#include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "cublas_routines.hpp"
#include "cuda_kernels.hpp"

void Fields::CheckSymmetries(){
#ifdef DEBUG
    if( current_step % 100 == 0) {
        std::printf("Computing divergence of v/B fields \n");
        ComputeDivergence();
    }
#endif
    if( current_step % param->symmetries_step) {

        CleanFieldDivergence();
        // if( current_step % 500*param->symmetries_step == 0) ComputeDivergence();
    }
    
}

void Fields::ComputeDivergence( ){

    double divvfield = 0.0;
    double divBfield = 0.0;
    int blocksPerGrid;
    cublasStatus_t stat;

    #ifdef INCOMPRESSIBLE
        // compute mean divergence for velocity field    
        blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
        Divergence<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *) wavevector.d_all_kvec, (data_type *) d_all_fields + ntotal_complex * VX, (data_type *)  d_tmparray[0], (size_t) ntotal_complex);
        // transform back to real space
        c2r_fft(d_tmparray[0], d_tmparray_r[0]);
        // compute absolute value of real vector (actually Dasum already does it...)
        // blocksPerGrid = ( 2 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
        // DoubleAbsolute<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *) d_tmparray_r[0], (scalar_type *) d_tmparray_r[0], (size_t) ntotal_complex);
        // reduce sum
        stat = cublasDasum(handle0, 2 * ntotal_complex, d_tmparray_r[0], 1, &divvfield);
        if (stat != CUBLAS_STATUS_SUCCESS) std::printf("-Reduce-sum of div v failed\n");
        std::printf("----Mean-divergence of v-field is %.2e\n",divvfield*param->lx/(2 * ntotal_complex));

         #ifdef MHD
            // compute mean divergence for magnetic field    
            blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
            Divergence<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *) wavevector.d_all_kvec, (data_type *) d_all_fields + ntotal_complex * BX, (data_type *)  d_tmparray[0], (size_t) ntotal_complex);
            // transform back to real space 
            c2r_fft(d_tmparray[0], d_tmparray_r[0]);
            // compute absolute value of real vector (actually Dasum already does it...)
            // blocksPerGrid = ( 2 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
            // DoubleAbsolute<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *) d_tmparray_r[0], (scalar_type *) d_tmparray_r[0], (size_t) ntotal_complex);
            // reduce sum
            stat = cublasDasum(handle0, 2 * ntotal_complex, d_tmparray_r[0], 1, &divBfield);
            if (stat != CUBLAS_STATUS_SUCCESS) std::printf("-Reduce-sum of div B failed\n");
            std::printf("----Mean-divergence of B-field is %.2e\n",divBfield*param->lx/(2 * ntotal_complex));
         #endif
    #endif

}

void Fields::CleanFieldDivergence( ){

    int blocksPerGrid;

    #ifdef INCOMPRESSIBLE
        // compute mean divergence for velocity field    
        blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
        // CleanDivergence<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *) wavevector.d_all_kvec, (cufftDoubleComplex *) d_all_fields + ntotal_complex * VX, (cufftDoubleComplex *) d_all_fields + ntotal_complex * VX, (size_t) ntotal_complex);
        CleanDivergence<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *) wavevector.d_all_kvec, (data_type *) d_all_fields + ntotal_complex * VX, (data_type *) d_all_fields + ntotal_complex * VX, (size_t) ntotal_complex);

         #ifdef MHD
            // compute mean divergence for magnetic field    
            blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
            // CleanDivergence<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *) wavevector.d_all_kvec, (cufftDoubleComplex *) d_all_fields + ntotal_complex * BX, (cufftDoubleComplex *) d_all_fields + ntotal_complex * BX, (size_t) ntotal_complex);
            CleanDivergence<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *) wavevector.d_all_kvec, (data_type *) d_all_fields + ntotal_complex * BX, (data_type *) d_all_fields + ntotal_complex * BX, (size_t) ntotal_complex);
            
         #endif
    #endif

}
