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

/**
* Insert here functions that remap outputs
* when doing shearing boxes
* before the
* fields are sent to host
* */

// void InputOutput::RemapOutput(scalar_type *RealVec) {
//
//     std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
//
//
//     grid.NX * grid.NY * (( grid.NZ / 2) + 1);
//
// }
//
//
// void TransposeYZ(scalar_type *RealVec){
//
//
// }
