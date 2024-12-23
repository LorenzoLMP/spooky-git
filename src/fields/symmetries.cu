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

    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    int current_step = supervisor_ptr->timestep_ptr->current_step;
    int symmetries_step = param_ptr->symmetries_step;
    double deltax = param_ptr->lx / grid.NX;
    double meanFieldDiv = 0.0;


    if (param_ptr->debug > 0 and current_step % 100 == 0) {
        std::printf("Computing divergence of v/B fields \n");
        if (param_ptr->incompressible) {
            // compute mean divergence for velocity field
            meanFieldDiv = ComputeDivergence(d_all_fields + grid.NTOTAL_COMPLEX * vars.VEL);
            std::printf("---- Mean-divergence of v-field is %.2e [< div Field> * Delta x]\n", meanFieldDiv*deltax);
        }
        if (param_ptr->mhd) {
            meanFieldDiv = ComputeDivergence(d_all_fields + grid.NTOTAL_COMPLEX * vars.MAG);
            std::printf("---- Mean-divergence of B-field is %.2e [< div Field> * Delta x]\n", meanFieldDiv*deltax);
        }
    }



    if( current_step % symmetries_step) {
        CleanFieldDivergence();


//         // clean divergence for velocity field
//         CleanFieldDivergence(d_all_fields + grid.NTOTAL_COMPLEX * vars.VEL);

//         // clean divergence for magnetic field
//         CleanFieldDivergence(d_all_fields + grid.NTOTAL_COMPLEX * vars.MAG);

    }
    
}

double Fields::ComputeDivergence( data_type* complex_Fields ){

    double FieldDiv = 0.0;
    // double divBfield = 0.0;
    int blocksPerGrid;
    cublasStatus_t stat;


    blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
    Divergence<<<blocksPerGrid, threadsPerBlock>>>(wavevector.d_all_kvec, complex_Fields, (data_type *)  d_tmparray[0], (size_t) grid.NTOTAL_COMPLEX);
    // transform back to real space
    c2r_fft(d_tmparray[0], d_tmparray_r[0]);
    // compute absolute value of real vector (actually Dasum already does it...)
    // reduce sum
    stat = cublasDasum(handle0, 2 * grid.NTOTAL_COMPLEX, d_tmparray_r[0], 1, &FieldDiv);
    if (stat != CUBLAS_STATUS_SUCCESS) std::printf("- Reduce-sum of ComputeDivergence failed\n");
    // std::printf("----Mean-divergence of v-field is %.2e / L\n",divvfield/(2 * grid.NTOTAL_COMPLEX));

    return FieldDiv/(2 * grid.NTOTAL_COMPLEX);

}

// void Fields::CleanDivergence(){
//

//         // clean divergence for velocity field
//         CleanFieldDivergence(d_all_fields + grid.NTOTAL_COMPLEX * vars.VEL);

//         // clean divergence for magnetic field
//         CleanFieldDivergence(d_all_fields + grid.NTOTAL_COMPLEX * vars.MAG);

//
// }

void Fields::CleanFieldDivergence( ){

    int blocksPerGrid;

    blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;


    if (param_ptr->incompressible) {
        CleanDivergence<<<blocksPerGrid, threadsPerBlock>>>(wavevector.d_all_kvec, d_all_fields + grid.NTOTAL_COMPLEX * vars.VEL, d_all_fields + grid.NTOTAL_COMPLEX * vars.VEL, (size_t) grid.NTOTAL_COMPLEX);
    }
    if (param_ptr->mhd) {
        CleanDivergence<<<blocksPerGrid, threadsPerBlock>>>(wavevector.d_all_kvec, d_all_fields + grid.NTOTAL_COMPLEX * vars.MAG, d_all_fields + grid.NTOTAL_COMPLEX * vars.MAG, (size_t) grid.NTOTAL_COMPLEX);
    }

}
