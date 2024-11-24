#include "define_types.hpp"
#include "physics.hpp"
#include "fields.hpp"
#include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "parameters.hpp"
#include "supervisor.hpp"

void Physics::AdvectTemperature(scalar_type *rfields_in, data_type *dfields_out) {

    int blocksPerGrid;
    std::shared_ptr<Fields> fields = supervisor->fields;
    std::shared_ptr<Parameters> param = supervisor->param;

    scalar_type* kvec = fields->wavevector.d_all_kvec;
    scalar_type* mask = fields->wavevector.d_mask;

    data_type* en_flux = fields->d_all_tmparray +  ntotal_complex * fields->num_fields;

#ifdef BOUSSINESQ
    // first compute energy flux vector [ u_x theta, u_y theta, u_z theta]
    // we can re-utilize tmparrays and store result in tmparray_r[num_fields] - tmparray_r[num_fields + 3]
    blocksPerGrid = ( 2 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    EnergyFluxVector<<<blocksPerGrid, threadsPerBlock>>>(rfields_in, (scalar_type *)en_flux,  2 * ntotal_complex);


    // take fourier transforms of the 3 energy flux vector components
    for (int n = fields->num_fields ; n < fields->num_fields + 3; n++) {
        r2c_fft((scalar_type*) en_flux + 2*n*ntotal_complex, en_flux + n*ntotal_complex, supervisor);
    }


    // compute derivative of energy flux vector and assign u nabla theta to the dfield for theta
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    NonLinBoussinesqAdv<<<blocksPerGrid, threadsPerBlock>>>(kvec, en_flux, dfields_out, mask, ntotal_complex);



#endif // Boussinesq


}
