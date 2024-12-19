#include "define_types.hpp"
#include "fields.hpp"
#include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "cublas_routines.hpp"
#include "cuda_kernels.hpp"
#include "parameters.hpp"
#include "timestepping.hpp"
#include "supervisor.hpp"

void Fields::CheckSymmetries(){
    int current_step = supervisor_ptr->timestep_ptr->current_step;
    int symmetries_step = supervisor_ptr->param_ptr->symmetries_step;
    double deltax = supervisor_ptr->param_ptr->lx / nx;
    double meanFielDiv = 0.0;
#ifdef DEBUG
    if( current_step % 100 == 0) {
        std::printf("Computing divergence of v/B fields \n");
#ifdef INCOMPRESSIBLE
        // compute mean divergence for velocity field
        meanFielDiv = ComputeDivergence(d_all_fields + ntotal_complex * VX);
        std::printf("---- Mean-divergence of v-field is %.2e [< div Field> * Delta x]\n", meanFielDiv*deltax);
#ifdef MHD
        meanFielDiv = ComputeDivergence(d_all_fields + ntotal_complex * BX);
        std::printf("---- Mean-divergence of B-field is %.2e [< div Field> * Delta x]\n", meanFielDiv*deltax);
#endif
#endif
    }
#endif
    if( current_step % symmetries_step) {

#ifdef INCOMPRESSIBLE
        // clean divergence for velocity field
        CleanFieldDivergence(d_all_fields + ntotal_complex * VX);
#ifdef MHD
        // clean divergence for magnetic field
        CleanFieldDivergence(d_all_fields + ntotal_complex * BX);
#endif
#endif
    }
    
}

double Fields::ComputeDivergence( data_type* complex_Fields ){

    double meanDivField = 0.0;
    // double divBfield = 0.0;
    int blocksPerGrid;
    cublasStatus_t stat;


    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    Divergence<<<blocksPerGrid, threadsPerBlock>>>(wavevector.d_all_kvec, complex_Fields, (data_type *)  d_tmparray[0], (size_t) ntotal_complex);
    // transform back to real space
    c2r_fft(d_tmparray[0], d_tmparray_r[0]);
    // compute absolute value of real vector (actually Dasum already does it...)
    // reduce sum
    stat = cublasDasum(handle0, 2 * ntotal_complex, d_tmparray_r[0], 1, &meanDivField);
    if (stat != CUBLAS_STATUS_SUCCESS) std::printf("- Reduce-sum of ComputeDivergence failed\n");
    // std::printf("----Mean-divergence of v-field is %.2e / L\n",divvfield/(2 * ntotal_complex));

    return meanDivField/(2 * ntotal_complex);

}

void Fields::CleanFieldDivergence( data_type* complex_Fields ){

    int blocksPerGrid;

    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    CleanDivergence<<<blocksPerGrid, threadsPerBlock>>>(wavevector.d_all_kvec, complex_Fields, complex_Fields, (size_t) ntotal_complex);

}
