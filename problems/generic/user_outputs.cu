// #include "define_types.hpp"
// // #include "fields.hpp"
// #include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
// #include "fields.hpp"
// #include "parameters.hpp"
#include "user_outputs.hpp"
#include "cublas_routines.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "cufft_routines.hpp"
#include "user_defined_cuda_kernels.hpp"

UserOutput::UserOutput() {
    // double lx, ly, lz;
    // read_Parameters();
}

UserOutput::~UserOutput() {
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
