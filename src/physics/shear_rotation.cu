#include "common.hpp"
#include "physics.hpp"
#include "fields.hpp"
#include "cufft_routines.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "parameters.hpp"
#include "supervisor.hpp"


double Physics::ShiftTime(double time) {

    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    double tremap;

    tremap = fmod( time + param_ptr->ly / (2.0 * param_ptr->shear * param_ptr->lx) , param_ptr->ly / (param_ptr->shear * param_ptr->lx)) - param_ptr->ly / (2.0 * param_ptr->shear * param_ptr->lx );

    return tremap;
}

