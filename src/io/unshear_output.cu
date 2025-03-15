#include "common.hpp"
#include <filesystem>
// #include <stdlib.h>
// #include "fields.hpp"
#include "cufft_routines.hpp"
// #include "../libs/HighFive/include/highfive/highfive.hpp"
#include <highfive/highfive.hpp>
#include "fields.hpp"
// #include "hdf5_io.hpp"
#include "parameters.hpp"
#include "inputoutput.hpp"
#include "timestepping.hpp"
#include "supervisor.hpp"
#include "cuda_kernels_generic.hpp"

/**
* Insert here functions that unshears outputs
* when doing shearing boxes
* before the
* fields are sent to host
* */

void InputOutput::UnshearOutput(data_type *AllComplexFields, scalar_type *AllFieldsRealTmp, double current_time) {

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    if (param_ptr->debug > 0){
        std::printf("Unshearing fields before output \n");
    }

    data_type *AllFieldsTmp = (data_type *) AllFieldsRealTmp;

    int blocksPerGrid = ( vars.NUM_FIELDS * grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;

    ComplexVecAssign<<<blocksPerGrid, threadsPerBlock>>>(AllComplexFields, AllFieldsTmp, vars.NUM_FIELDS * grid.NTOTAL_COMPLEX);

    for (int n = 0 ; n < vars.NUM_FIELDS ; n++) {
        UnshearField(AllFieldsTmp + n*grid.NTOTAL_COMPLEX, current_time);
    }

    Complex2RealFields(AllFieldsTmp, vars.NUM_FIELDS);

}

void InputOutput::UnshearField(data_type *vecField, double current_time) {

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    scalar_type *ky = fields_ptr->wavevector.d_kvec[vars.KY];
    // this is the velocity of the boundaries w.r.t. x=0, it is necessary
    // so that x=0 does not move in space
    double tvelocity = fmod(current_time, 2.0 * param_ptr->ly / (param_ptr->shear * param_ptr->lx));

    int blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;

    UnshearComplexVec<<<blocksPerGrid, threadsPerBlock>>>(vecField, ky, tvelocity*param_ptr->lx*param_ptr->shear, grid.NTOTAL_COMPLEX);



}
//
//
// void TransposeYZ(scalar_type *RealVec){
//
//
// }
