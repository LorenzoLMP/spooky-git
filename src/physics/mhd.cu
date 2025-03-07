#include "common.hpp"
#include "physics.hpp"
// #include "timestepping.hpp"
// #include "cufft_routines.hpp"
#include "cublas_routines.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "parameters.hpp"
// #include "inputoutput.hpp"
#include "fields.hpp"
#include <cuda_runtime.h>
// #include <cufftXt.h>
#include "cufft_utils.h"
#include "cufft_routines.hpp"
#include "supervisor.hpp"
// #include "timestepping.hpp"
// #include "rkl.hpp"


void Physics::CurlEMF(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dMag) {

    /*
    *
    * Here we do the curl of the emf
    *
    */

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    int blocksPerGrid;

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;
    scalar_type* mask = fields_ptr->wavevector.d_mask;

    if (param_ptr->mhd) {

        scalar_type* real_velField = real_Buffer + vars.VEL * 2 * grid.NTOTAL_COMPLEX ;
        scalar_type* real_magField = real_Buffer + vars.MAG * 2 * grid.NTOTAL_COMPLEX ;

        // compute emf = u x B:
        // emf_x = u_y B_z - u_z B_y , emf_y = u_z B_x - u_x B_z , emf_z = u_x B_y - u_y B_x
        // the results are saved in the first 3 temp_arrays as [emf_x, emf_y, emf_z] (they are the x,y,z components of the emf)
        // We can re-utilize tmparrays and store result in in the temp_arrays from [0, 1, 2]

        data_type* emf = fields_ptr->d_all_tmparray;

        blocksPerGrid = ( 2 * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        MagneticEmf<<<blocksPerGrid, threadsPerBlock>>>(real_velField, real_magField, (scalar_type*) emf,  2 * grid.NTOTAL_COMPLEX);

        // take fourier transforms of the 3 independent components of the antisymmetric shear matrix
        for (int n = 0; n < 3; n++) {
            r2c_fft((scalar_type*) emf + 2*n*grid.NTOTAL_COMPLEX, emf + n*grid.NTOTAL_COMPLEX, supervisor_ptr);
        }

        // compute derivative of antisymmetric magnetic shear matrix and assign to dfields

        blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;
        MagneticShear<<<blocksPerGrid, threadsPerBlock>>>(kvec, emf, complex_dMag, mask, grid.NTOTAL_COMPLEX);
    }

}