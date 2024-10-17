#include "define_types.hpp"
#include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "timestepping.hpp"
#include "parameters.hpp"
#include "fields.hpp"
#include "physics.hpp"

void TimeStepping::compute_dfield(Fields &fields, Parameters &param, Physics &phys) {
    NVTX3_FUNC_RANGE();

    int blocksPerGrid;
    /*
     * Do all computations
     * required to compute dfield
     *
     */
#ifdef DDEBUG
    std::printf("Now entering compute_dfield function \n");
#endif


#ifdef HEAT_EQ

    // dT = nu_th nabla T
    //   #ifndef ANISOTROPIC_DIFFUSION
        // dfldo.th[i] += - nu_th * k2t[i] * fldi.th[i];
    //   #endif
    // computes nabla operator of T and assigns to dT
    // int blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    // nablaOp<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *) wavevector.d_kvec[KX],  (scalar_type *) wavevector.d_kvec[KY], (scalar_type *) wavevector.d_kvec[KZ], (cufftDoubleComplex *) d_farray[TH], (cufftDoubleComplex *) d_dfarray[TH], param.nu_th, (size_t) ntotal_complex, ASS);

    if (stage_step == 0) compute_dt(fields, param, phys);

    // laplacianScalar((scalar_type **)wavevector.d_kvec, (cufftDoubleComplex *) d_farray[TH], (cufftDoubleComplex *) d_dfarray[TH], param.nu_th, ASS);
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    // nablaOp<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *) wavevector.d_kvec[KX],  (scalar_type *) wavevector.d_kvec[KY], (scalar_type *) wavevector.d_kvec[KZ], (cufftDoubleComplex *) d_farray[TH], (cufftDoubleComplex *) d_dfarray[TH], param.nu_th, (size_t) ntotal_complex, ASS);
    nablaOpScalar<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *) fields.wavevector.d_all_kvec, (data_type *) fields.d_farray[TH], (data_type *) fields.d_dfarray[TH], param.nu_th, (size_t) ntotal_complex, ASS);

#endif

    // advectFields((scalar_type **)wavevector.d_kvec, (cufftDoubleComplex **) d_farray, (cufftDoubleComplex **) d_dfarray, ASS);

    // we are assuming that the first 3 arrays are always vx vy vz
    // for (int n = 0 ; n < num_fields ; n++) {
    //     c2r_fft(array_input[n], scratch[n]);
    // }
#ifdef INCOMPRESSIBLE

     // assign fields to [num_fields] tmparray (memory block starts at d_all_tmparray)
    blocksPerGrid = ( fields.num_fields * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    ComplexVecAssign<<<blocksPerGrid, threadsPerBlock>>>((cufftDoubleComplex *)fields.d_all_fields, (cufftDoubleComplex *)fields.d_all_tmparray, fields.num_fields * ntotal_complex);

    // compute FFTs from complex to real fields to start computation of shear traceless matrix
    for (int n = 0; n < fields.num_fields; n++){
        c2r_fft(fields.d_tmparray[n], fields.d_tmparray_r[n], supervisor);
    }

    // cudaDeviceSynchronize();
    if (stage_step == 0) compute_dt(fields, param, phys);

    // we use Basdevant formulation [1983]
    // compute the elements of the traceless symmetric matrix B_ij = u_i u_j - delta_ij Tr (u_i u_j) / 3. It has only 5 independent components B_xx, B_xy, B_xz, Byy, B_yz. (B_zz = - B_xx - B_yy)
    // the results are saved in the temp_arrays from [num_fields -- num_fields + 5] (the first num_fields arrays are reserved for the real-valued fields)
    blocksPerGrid = ( 2 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
#ifndef MHD
    TracelessShearMatrix<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)fields.d_all_tmparray, (scalar_type *)fields.d_all_tmparray + 2 * ntotal_complex * fields.num_fields,  2 * ntotal_complex);
#else
    TracelessShearMatrixMHD<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)fields.d_all_tmparray, (scalar_type *)fields.d_all_tmparray + 2 * ntotal_complex * fields.num_fields,  2 * ntotal_complex);
#endif


    // take fft of 5 independent components of B_ij
    for (int n = fields.num_fields ; n < fields.num_fields + 5; n++) {
        r2c_fft(fields.d_tmparray_r[n], fields.d_tmparray[n], supervisor);
    }

    // compute derivative of traceless shear matrix and assign to dfields
    // this kernel works also if MHD
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    NonLinHydroAdv<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)fields.wavevector.d_all_kvec, (data_type *)fields.d_all_tmparray + ntotal_complex * fields.num_fields, (data_type *) fields.d_all_dfields, (scalar_type *)fields.wavevector.d_mask, ntotal_complex);


#ifdef MHD
    // compute emf = u x B:
    // emf_x = u_y B_z - u_z B_y , emf_y = u_z B_x - u_x B_z , emf_z = u_x B_y - u_y B_x
    // the results are saved in the first 3 temp_arrays (after those reserved for the fields, the memory block points already at the right location) as [emf_x, emf_y, emf_z] (they are the x,y,z components of the emf)
    // We can re-utilize tmparrays and store result in tmparray_r[num_fields] - tmparray_r[num_fields + 3]
    blocksPerGrid = ( 2 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    MagneticEmf<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)fields.d_all_tmparray, (scalar_type *)fields.d_all_tmparray + 2 * ntotal_complex * fields.num_fields,  2 * ntotal_complex);

    // take fourier transforms of the 3 independent components of the antisymmetric shear matrix
    for (int n = fields.num_fields ; n < fields.num_fields + 3; n++) {
        r2c_fft(fields.d_tmparray_r[n], fields.d_tmparray[n], supervisor);
    }

    // compute derivative of antisymmetric magnetic shear matrix and assign to dfields
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    MagneticShear<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)fields.wavevector.d_all_kvec, (data_type *)fields.d_all_tmparray + ntotal_complex * fields.num_fields, (data_type *) fields.d_all_dfields, (scalar_type *)fields.wavevector.d_mask, ntotal_complex);



#endif

#ifdef BOUSSINESQ
    // This function assumes that the real transforms of the fields are stored in tmparrays_r[0] - tmparray_r[num_fields - 1]
    phys.Boussinesq(fields, param);
#endif

// #ifdef BOUSSINESQ
//     // for hydro-Boussinesq computation of u nabla theta can go here
//
//     // first compute energy flux vector [ u_x theta, u_y theta, u_z theta]
//     // we can re-utilize tmparrays and store result in tmparray_r[num_fields] - tmparray_r[num_fields + 3]
//     blocksPerGrid = ( 2 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
//     EnergyFluxVector<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)d_all_tmparray, (scalar_type *)d_all_tmparray + 2 * ntotal_complex * num_fields,  2 * ntotal_complex);
//
//     // scalar_type *host_tmp;
//     // host_tmp = (scalar_type *) malloc( (size_t) sizeof(scalar_type) * 2 * ntotal_complex );
//     // for (int i = 0; i < 2 * ntotal_complex ; i++){
//     //     host_tmp[i] = 0.0;
//     // }
//     // CUDA_RT_CALL(cudaMemcpy(host_tmp, (scalar_type*)d_tmparray_r[4], sizeof(scalar_type) * 2 * ntotal_complex , cudaMemcpyDeviceToHost));
//     // unsigned int idx;
//     // for (int i = 25; i < 32; i++){
//     //     idx =  (nz/2+1)*2 * ( i * ny);
//     //     // std::printf("v1[%d]= %f \t v2[%d]= %f \n", idx, farray_r[0][idx], idx, farray_r[1][idx]);
//     //     for (int n = 0; n < 1; n++){
//     //         std::printf("tmp[%d][%d] = %.3e \t", n, idx, host_tmp[idx]);
//     //     }
//     //     std::cout << std::endl;
//     // }
//     // for (int i = 0; i < 2 * ntotal_complex ; i++){
//     //     if (host_tmp[i] != 0.0) {
//     //         std::printf("BREAK:   tmp[%d] = %.3e \n", i, host_tmp[i]);
//     //         break;
//     //     }
//     // }
//     // free(host_tmp);
//
//     // take fourier transforms of the 3 energy flux vector components
//     for (int n = num_fields ; n < num_fields + 3; n++) {
//         r2c_fft(d_tmparray_r[n], d_tmparray[n]);
//     }
//
//     // scalar_type *host_tmp;
//     // host_tmp = (scalar_type *) malloc( (size_t) sizeof(scalar_type) * 2 * ntotal_complex );
//     // for (int i = 0; i < 2 * ntotal_complex ; i++){
//     //     host_tmp[i] = 0.0;
//     // }
//     // CUDA_RT_CALL(cudaMemcpy(host_tmp, (scalar_type*)d_tmparray[4], sizeof(scalar_type) * 2 * ntotal_complex , cudaMemcpyDeviceToHost));
//     // unsigned int idx;
//     // for (int i = 0; i < 25; i++){
//     //     idx =  (nz/2+1)*2 * ( i * ny);
//     //     // std::printf("v1[%d]= %f \t v2[%d]= %f \n", idx, farray_r[0][idx], idx, farray_r[1][idx]);
//     //     for (int n = 0; n < 1; n++){
//     //         std::printf("tmp[%d][%d] = %.3e \t", n, idx, host_tmp[idx]);
//     //     }
//     //     std::cout << std::endl;
//     // }
//     // free(host_tmp);
//
//     // compute derivative of energy flux vector and assign u nabla theta to the dfield for theta
//     blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
//     NonLinBoussinesqAdv<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)wavevector.d_all_kvec, (data_type *)d_all_tmparray + ntotal_complex * num_fields, (data_type *) d_all_dfields, (scalar_type *)wavevector.d_mask, ntotal_complex);
//
//
//
// #ifdef STRATIFICATION
//     EntropyStratification();
//     // add - th e_strat to velocity component in the strat direction
//     // add N2 u_strat to temperature equation
//     // this is for normalization where theta is in units of g [L/T^2]
//     // other normalizations possible
//     // blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
//     // BoussinesqStrat<<<blocksPerGrid, threadsPerBlock>>>( (data_type *)d_all_fields, (data_type *) d_all_dfields, param.N2, ntotal_complex, STRAT_DIR);
// #endif
//
// #ifndef ANISOTROPIC_DIFFUSION
//     //  for explicit treatment of energy diffusion term
//     blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
//     nablaOpScalar<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *) wavevector.d_all_kvec, (data_type *) d_farray[TH], (data_type *) d_dfarray[TH], param.nu_th, (size_t) ntotal_complex, ADD);
// #else
// #ifdef MHD
//     AnisotropicConduction();
//     /*
//     // assign Bx, By, Bz to first 3 scratch arrays
//     blocksPerGrid = ( 3 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
//     ComplexVecAssign<<<blocksPerGrid, threadsPerBlock>>>((cufftDoubleComplex *)d_all_fields + ntotal_complex * BX, (cufftDoubleComplex *)d_all_tmparray, 3 * ntotal_complex);
//     // compute gradient of theta and assign it to next 3 scratch arrays
//     blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
//     Gradient<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)wavevector.d_all_kvec, (data_type *) d_farray[TH], (data_type *)d_all_tmparray + 3 * ntotal_complex, ntotal_complex);
//     // compute complex to real iFFTs
//     for (int n = 0; n < 6; n++){
//         c2r_fft(d_tmparray[n], d_tmparray_r[n]);
//     }
//     // compute the scalar B grad theta (real space) and assign it to 7th scratch array
//     blocksPerGrid = ( 2 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
//     ComputeBGradTheta<<<blocksPerGrid, threadsPerBlock>>>( (scalar_type *) d_tmparray_r[0], (scalar_type *) d_tmparray_r[3], (scalar_type *) d_tmparray_r[6], 2 * ntotal_complex);
//     // compute the anisotropic heat flux and put it in the 3-4-5 tmp arrays
//     ComputeAnisotropicHeatFlux<<<blocksPerGrid, threadsPerBlock>>>( (scalar_type *) d_tmparray_r[0], (scalar_type *) d_tmparray_r[6], (scalar_type *) d_tmparray_r[3], param.OmegaT2, (1./param.reynolds_ani), 2 * ntotal_complex, STRAT_DIR);
//     // take fourier transforms of the heat flux
//     for (int n = 3 ; n < 6; n++) {
//         r2c_fft(d_tmparray_r[n], d_tmparray[n]);
//     }
//     // take divergence of heat flux
//     blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
//     DivergenceMask<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)wavevector.d_all_kvec, (data_type *) d_tmparray[3], (data_type *) d_all_dfields + TH * ntotal_complex, (scalar_type *)wavevector.d_mask, ntotal_complex, ADD);
//     */
// #endif   // MHD
// #endif   // ANISOTROPIC_DIFFUSION
//
//
// #endif // Boussinesq

/*
 *
 * Now we enforce the incompressibility
 * condition
 *
 */


    // compute pseudo-pressure and subtract grad p_tilde from dfields
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    GradPseudoPressure<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *)fields.wavevector.d_all_kvec, (data_type *) fields.d_all_dfields, ntotal_complex);


/*
 *
 * Here we do the diffusion terms
 *
 */

    // for explicit treatment of diffusion terms
    // with incompressible d_all_fields always points at VX
    blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    nablaOpVector<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *) fields.wavevector.d_all_kvec, (data_type *) fields.d_all_fields, (data_type *) fields.d_all_dfields, param.nu, (size_t) ntotal_complex, ADD);

#ifdef MHD
    // for explicit treatment of diffusion terms
    // point d_all_fields at BX
    nablaOpVector<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *) fields.wavevector.d_all_kvec, (data_type *) fields.d_all_fields + ntotal_complex * BX, (data_type *) fields.d_all_dfields + ntotal_complex * BX, param.nu_m, (size_t) ntotal_complex, ADD);
#endif



    // will use temp arrays to store data d_tmparray_r and d_tmparray
    // for (int n = 0 ; n < num_fields ; n++) {
    //     c2r_fft(d_farray[n], d_tmparray_r[n]);
    // }
    //
    //
    //
    // for (int n = 0 ; n < num_fields ; n++) {
    //     r2c_fft(d_farray_r[n], d_farray[n]);
    // }


#endif //end INCOMPRESSIBLE



}


