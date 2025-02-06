#include "common.hpp"
#include "timestepping.hpp"
#include "cufft_routines.hpp"
#include "hydro_mhd_advance.hpp"
#include "cublas_routines.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "parameters.hpp"
#include "fields.hpp"
#include "supervisor.hpp"
#include "rkl.hpp"
#include "physics.hpp"


void TimeStepping::ShiftTime() {

    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    if (param_ptr->debug > 0){
        std::printf("Computing remap time \n");
    }

    tremap = fmod( current_time + param_ptr->ly / (2.0 * param_ptr->shear * param_ptr->lx) , param_ptr->ly / (param_ptr->shear * param_ptr->lx)) - param_ptr->ly / (2.0 * param_ptr->shear * param_ptr->lx );

}

void TimeStepping::RemapField(data_type *vecField) {

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    if (param_ptr->debug > 0){
        std::printf("Remapping field \n");
    }

    data_type *vecRemap = fields_ptr->d_tmparray[0];
    scalar_type *mask = fields_ptr->wavevector.d_mask;

    int blocksPerGrid = ( grid.NTOTAL_COMPLEX + threadsPerBlock - 1) / threadsPerBlock;

    VecInitComplex<<<blocksPerGrid, threadsPerBlock>>>(vecRemap, data_type(0.0,0.0), grid.NTOTAL_COMPLEX);

    RemapComplexVec<<<blocksPerGrid, threadsPerBlock>>>(vecField, vecRemap, grid.NX, grid.NY, grid.NZ, grid.NTOTAL_COMPLEX);

    MaskVector<<<blocksPerGrid, threadsPerBlock>>>(vecRemap, mask, vecField, grid.NTOTAL_COMPLEX);
}

void TimeStepping::RemapAllFields(data_type *AllComplexFields){

    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    if (param_ptr->debug > 0){
        std::printf("Remapping all fields \n");
    }

    for (int n = 0 ; n < vars.NUM_FIELDS ; n++) {
        RemapField(AllComplexFields + n*grid.NTOTAL_COMPLEX);
    }
}
