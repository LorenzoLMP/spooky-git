#include "define_types.hpp"
#include "physics.hpp"
#include "fields->hpp"
#include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "parameters.hpp"
#include "supervisor.hpp"

void Physics::Boussinesq() {

    int blocksPerGrid;
    std::shared_ptr<Fields> fields = supervisor->fields;
    std::shared_ptr<Parameters> param = supervisor->param;

#ifdef BOUSSINESQ
    // first compute energy flux vector [ u_x theta, u_y theta, u_z theta]
    // we can re-utilize tmparrays and store result in tmparray_r[num_fields] - tmparray_r[num_fields + 3]
    blocksPerGrid = ( 2 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    EnergyFluxVector<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)fields->d_all_tmparray, (scalar_type *)fields->d_all_tmparray + 2 * ntotal_complex * fields->num_fields,  2 * ntotal_complex);


    // take fourier transforms of the 3 energy flux vector components
    for (int n = fields->num_fields ; n < fields->num_fields + 3; n++) {
        r2c_fft(fields->d_tmparray_r[n], fields->d_tmparray[n], supervisor);
    }


    // compute derivative of energy flux vector and assign u nabla theta to the dfield for theta
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    NonLinBoussinesqAdv<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)fields->wavevector.d_all_kvec, (data_type *)fields->d_all_tmparray + ntotal_complex * fields->num_fields, (data_type *) fields->d_all_dfields, (scalar_type *)fields->wavevector.d_mask, ntotal_complex);



#endif // Boussinesq


}
