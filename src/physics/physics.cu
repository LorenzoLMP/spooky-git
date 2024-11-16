#include "define_types.hpp"
#include "physics.hpp"
// #include "timestepping.hpp"
// #include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "cublas_routines.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "parameters.hpp"
// #include "inputoutput.hpp"
#include "fields.hpp"
#include <cuda_runtime.h>
// #include <cufftXt.h>
// #include "spooky.hpp"
#include "cufft_utils.h"
#include "cufft_routines.hpp"
// #include "define_types.hpp"
#include "supervisor.hpp"


Physics::Physics(Supervisor &sup_in){

    supervisor = &sup_in;

}

Physics::~Physics(){

}

void Physics::HyperbolicTerms(){
    /*
    *
    * Here we do the hyperbolic terms
    *
    */
#ifdef DDEBUG
    std::printf("Now entering compute_parabolic_terms function \n");
#endif

    std::shared_ptr<Fields> fields = supervisor->fields;
    std::shared_ptr<Parameters> param = supervisor->param;

    int blocksPerGrid;

#ifdef INCOMPRESSIBLE
    // we use Basdevant formulation [1983]
    // compute the elements of the traceless symmetric matrix B_ij = u_i u_j - delta_ij Tr (u_i u_j) / 3. It has only 5 independent components B_xx, B_xy, B_xz, Byy, B_yz. (B_zz = - B_xx - B_yy)
    // the results are saved in the temp_arrays from [num_fields -- num_fields + 5] (the first num_fields arrays are reserved for the real-valued fields)
    blocksPerGrid = ( 2 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
#ifndef MHD
    TracelessShearMatrix<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)fields->d_all_tmparray, (scalar_type *)fields->d_all_tmparray + 2 * ntotal_complex * fields->num_fields,  2 * ntotal_complex);
#else
    TracelessShearMatrixMHD<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)fields->d_all_tmparray, (scalar_type *)fields->d_all_tmparray + 2 * ntotal_complex * fields->num_fields,  2 * ntotal_complex);
#endif


    // take fft of 5 independent components of B_ij
    for (int n = fields->num_fields ; n < fields->num_fields + 5; n++) {
        r2c_fft(fields->d_tmparray_r[n], fields->d_tmparray[n], supervisor);
    }

    // compute derivative of traceless shear matrix and assign to dfields
    // this kernel works also if MHD
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    NonLinHydroAdv<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)fields->wavevector.d_all_kvec, (data_type *)fields->d_all_tmparray + ntotal_complex * fields->num_fields, (data_type *) fields->d_all_dfields, (scalar_type *)fields->wavevector.d_mask, ntotal_complex);


#ifdef MHD
    // compute emf = u x B:
    // emf_x = u_y B_z - u_z B_y , emf_y = u_z B_x - u_x B_z , emf_z = u_x B_y - u_y B_x
    // the results are saved in the first 3 temp_arrays (after those reserved for the fields, the memory block points already at the right location) as [emf_x, emf_y, emf_z] (they are the x,y,z components of the emf)
    // We can re-utilize tmparrays and store result in tmparray_r[num_fields] - tmparray_r[num_fields + 3]
    blocksPerGrid = ( 2 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    MagneticEmf<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)fields->d_all_tmparray, (scalar_type *)fields->d_all_tmparray + 2 * ntotal_complex * fields->num_fields,  2 * ntotal_complex);

    // take fourier transforms of the 3 independent components of the antisymmetric shear matrix
    for (int n = fields->num_fields ; n < fields->num_fields + 3; n++) {
        r2c_fft(fields->d_tmparray_r[n], fields->d_tmparray[n], supervisor);
    }

    // compute derivative of antisymmetric magnetic shear matrix and assign to dfields
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    MagneticShear<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)fields->wavevector.d_all_kvec, (data_type *)fields->d_all_tmparray + ntotal_complex * fields->num_fields, (data_type *) fields->d_all_dfields, (scalar_type *)fields->wavevector.d_mask, ntotal_complex);

#endif

#ifdef BOUSSINESQ
    // does the advection of the temperature
    // This function assumes that the real transforms of the fields are stored in tmparrays_r[0] - tmparray_r[num_fields - 1]
    Boussinesq();
#endif

#endif // INCOMPRESSIBLE
}

void Physics::ParabolicTerms(data_type *fields_in, data_type *dfields_out){
    /*
    *
    * Here we do the diffusion terms
    *
    */
#ifdef DDEBUG
    std::printf("Now entering compute_parabolic_terms function \n");
#endif

    std::shared_ptr<Fields> fields = supervisor->fields;
    std::shared_ptr<Parameters> param = supervisor->param;

    int blocksPerGrid;

    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;

#ifdef INCOMPRESSIBLE
    // for explicit treatment of diffusion terms
    // with incompressible d_all_fields always points at VX
    nablaOpVector<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *) fields->wavevector.d_all_kvec, (data_type *) fields_in + ntotal_complex * VX, (data_type *) dfields_out + ntotal_complex * VX, param->nu, (size_t) ntotal_complex, ADD);

#ifdef MHD
    // for explicit treatment of diffusion terms
    // point d_all_fields at BX
    nablaOpVector<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *) fields->wavevector.d_all_kvec, (data_type *) fields_in + ntotal_complex * BX, (data_type *) dfields_out + ntotal_complex * BX, param->nu_m, (size_t) ntotal_complex, ADD);
#endif

#endif

#if defined(HEAT_EQ) || defined(BOUSSINESQ)
#ifndef ANISOTROPIC_DIFFUSION
    //  for explicit treatment of energy diffusion term
    // blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;

#ifdef HEAT_EQ
    // this is because the nabla scalar will *add* to d_dfarray, and with HEAT_EQ we want to *set*
    VecInitComplex<<<blocksPerGrid, threadsPerBlock>>>((data_type *)dfields_out + ntotal_complex * TH, data_type(0.0,0.0), ntotal_complex);
#endif

    nablaOpScalar<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *) fields->wavevector.d_all_kvec, (data_type *) fields_in + ntotal_complex * TH, (data_type *) dfields_out + ntotal_complex * TH, param->nu_th, (size_t) ntotal_complex, ADD);
#else
#ifdef MHD
// #ifndef SUPERTIMESTEPPING
    // AnisotropicConduction(fields, param);
    // this function requires that the real fields are saved in the first num_fields temp arrays
    AnisotropicConduction((data_type *) fields_in + ntotal_complex * TH, (data_type *) dfields_out + ntotal_complex * TH);
// #endif

#endif   // MHD
#endif   // ANISOTROPIC_DIFFUSION

#endif // defined(HEAT_EQ) || defined(BOUSSINESQ)
}
