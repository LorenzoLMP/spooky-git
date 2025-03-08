#include "common.hpp"
// #include "parameters.hpp"
// #include "user_outputs.hpp"
#include "cublas_routines.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "cufft_routines.hpp"
#include "user_defined_cuda_kernels.hpp"

#include "fields.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include "parameters.hpp" //includes user_outputs
#include "inputoutput.hpp"
#include "timestepping.hpp"
#include "supervisor.hpp"

UserOutput::UserOutput(Supervisor &sup_in)
    : SpookyOutput(sup_in) {
    // double lx, ly, lz;
    // read_Parameters();
    // supervisor_ptr = &sup_in;
}

// UserOutput::~UserOutput() {
// }



scalar_type UserOutput::computekpartheta(data_type* complex_Fields,
                                        scalar_type* real_Buffer){

    scalar_type average_kpartheta = 0.0;
    int blocksPerGrid;

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    // std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;
    // scalar_type* mask = fields_ptr->wavevector.d_mask;
    data_type* complex_Theta = complex_Fields + vars.TH * grid.NTOTAL_COMPLEX ;
    data_type* grad_theta = fields_ptr->d_tmparray[0];
    
    blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
    Gradient<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_Theta, grad_theta, grid.NTOTAL_COMPLEX);

    // compute complex to real iFFTs
    for (int n = 0; n < 3; n++){
        c2r_fft(grad_theta + n*grid.NTOTAL_COMPLEX, ((scalar_type *) grad_theta) + 2*n*grid.NTOTAL_COMPLEX );
    }

    // compute the scalar b grad theta (real space) and assign it to [3] scratch array
    scalar_type* real_magField = real_Buffer + 2 * grid.NTOTAL_COMPLEX * vars.MAG;
    scalar_type* bgrad_theta = fields_ptr->d_tmparray_r[3];

    blocksPerGrid = ( 2 * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
    ComputeBGradTheta<<<blocksPerGrid, threadsPerBlock>>>(real_magField, (scalar_type *) grad_theta, bgrad_theta, 2 * grid.NTOTAL_COMPLEX);

    scalar_type theta_2 = 2.*computeEnergy(complex_Theta);

    average_kpartheta = twoFieldCorrelation(bgrad_theta, bgrad_theta) / theta_2;

    return average_kpartheta;
}



scalar_type UserOutput::customFunction( data_type *vcomplex ) {
    /***
     * This function uses complex input to compute the "energy"
     * The modes with k>0 only have half the energy (because the k<0 is not present).
     * Here we multiply all k modes by 2 and then subtract once the energy in the k=0 mode.
     * The total is then divided by 2 to give quantity (i.e. Energy ~ (1/2) v^2)
     ***/

    // cublasStatus_t stat;
    // scalar_type norm = 0.0;
    scalar_type var = 0.0;
    // scalar_type subtract = 0.0;
    // scalar_type tmp = 0.0;

    return var;
}
