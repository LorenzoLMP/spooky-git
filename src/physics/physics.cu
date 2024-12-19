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

    supervisor_ptr = &sup_in;

}

Physics::~Physics(){

}

void Physics::HyperbolicTerms(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dFields){
    /*
    *
    * Here we do the hyperbolic terms
    *
    */
#ifdef DDEBUG
    std::printf("Now entering HyperbolicTerms function \n");
#endif

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    int blocksPerGrid;

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;
    scalar_type* mask = fields_ptr->wavevector.d_mask;

#ifdef INCOMPRESSIBLE

    // we use Basdevant formulation [1983]
    // compute the elements of the traceless symmetric matrix
    // B_ij = u_i u_j - delta_ij Tr (u_i u_j) / 3.
    // It has only 5 independent components B_xx, B_xy, B_xz, Byy, B_yz.
    // (B_zz = - B_xx - B_yy)
    // The results are saved in the temp_arrays from [0, 1, ..., 4]
    scalar_type* shear_matrix = fields_ptr->d_all_tmparray;

    blocksPerGrid = ( 2 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
#ifndef MHD
    TracelessShearMatrix<<<blocksPerGrid, threadsPerBlock>>>(real_Buffer, shear_matrix,  2 * ntotal_complex);
#else
    TracelessShearMatrixMHD<<<blocksPerGrid, threadsPerBlock>>>(real_Buffer, shear_matrix,  2 * ntotal_complex);
#endif


    // take fft of 5 independent components of B_ij
    for (int n = 0; n < 5; n++) {
        r2c_fft(shear_matrix + 2*n*ntotal_complex, ((data_type*) shear_matrix) + n*ntotal_complex, supervisor);
    }

    // compute derivative of traceless shear matrix and assign to dfields
    // this kernel works also if MHD
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    NonLinHydroAdv<<<blocksPerGrid, threadsPerBlock>>>(kvec, (data_type*) shear_matrix, complex_dFields, mask, ntotal_complex);


#ifdef MHD
    // compute emf = u x B:
    // emf_x = u_y B_z - u_z B_y , emf_y = u_z B_x - u_x B_z , emf_z = u_x B_y - u_y B_x
    // the results are saved in the first 3 temp_arrays as [emf_x, emf_y, emf_z] (they are the x,y,z components of the emf)
    // We can re-utilize tmparrays and store result in in the temp_arrays from [0, 1, 2]

    scalar_type* emf = fields_ptr->d_all_tmparray;

    blocksPerGrid = ( 2 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    MagneticEmf<<<blocksPerGrid, threadsPerBlock>>>(real_Buffer, emf,  2 * ntotal_complex);

    // take fourier transforms of the 3 independent components of the antisymmetric shear matrix
    for (int n = 0; n < 3; n++) {
        r2c_fft(emf + 2*n*ntotal_complex, ((data_type*) emf) + n*ntotal_complex, supervisor);
    }

    // compute derivative of antisymmetric magnetic shear matrix and assign to dfields
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    MagneticShear<<<blocksPerGrid, threadsPerBlock>>>(kvec, (data_type *)emf, complex_dFields, mask, ntotal_complex);

#endif

#ifdef BOUSSINESQ
    // does the advection of the temperature
    AdvectTemperature(complex_Fields, real_Buffer, complex_dFields);
#endif

#endif // INCOMPRESSIBLE
}


void Physics::ParabolicTerms(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dFields){
    /*
    *
    * Here we do the diffusion terms
    *
    */
#ifdef DDEBUG
    std::printf("Now entering ParabolicTerms function \n");
#endif

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

    scalar_type* kvec = fields_ptr->wavevector.d_all_kvec;

    int blocksPerGrid;

    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;

#ifdef INCOMPRESSIBLE
    // for explicit treatment of diffusion terms
    // with incompressible d_all_fields always points at VX
    nablaOpVector<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_Fields + ntotal_complex * VX, complex_dFields + ntotal_complex * VX, param_ptr->nu, (size_t) ntotal_complex, ADD);

#ifdef MHD
    // for explicit treatment of diffusion terms
    // point d_all_fields at BX
    nablaOpVector<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_Fields + ntotal_complex * BX, complex_dFields + ntotal_complex * BX, param_ptr->nu_m, (size_t) ntotal_complex, ADD);
#endif

#endif

#if defined(HEAT_EQ) || defined(BOUSSINESQ)
#ifndef ANISOTROPIC_DIFFUSION
    //  for explicit treatment of energy diffusion term
    // blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;

#ifdef HEAT_EQ
    // this is because the nabla scalar will *add* to d_dfarray, and with HEAT_EQ we want to *set*
    VecInitComplex<<<blocksPerGrid, threadsPerBlock>>>(complex_dFields + ntotal_complex * TH, data_type(0.0,0.0), ntotal_complex);
#endif

    nablaOpScalar<<<blocksPerGrid, threadsPerBlock>>>(kvec, complex_Fields + ntotal_complex * TH, complex_dFields + ntotal_complex * TH, param_ptr->nu_th, (size_t) ntotal_complex, ADD);
#else
#ifdef MHD
// #ifndef SUPERTIMESTEPPING
    // AnisotropicConduction(fields, param);
    AnisotropicConduction(complex_Fields, real_Buffer, complex_dFields + ntotal_complex * TH);
// #endif

#endif   // MHD
#endif   // ANISOTROPIC_DIFFUSION

#endif // defined(HEAT_EQ) || defined(BOUSSINESQ)
}
