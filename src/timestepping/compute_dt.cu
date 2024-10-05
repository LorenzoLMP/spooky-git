#include "define_types.hpp"
#include "timestepping.hpp"
#include "fields.hpp"
#include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "cublas_routines.hpp"
#include "parameters.hpp"
#include "physics.hpp"

// we are assuming that the fields have been already fft to real and saved in d_tmparray_r
void TimeStepping::compute_dt(Fields &fields, Parameters &param, Physics &phys) {

    NVTX3_FUNC_RANGE();
    double dt;

#ifdef DDEBUG
    std::printf("Now entering compute_dt function \n");
#endif

#ifdef INCOMPRESSIBLE
    double gamma_v = 0.0, gamma_tot = 0.0;
    double maxfx, maxfy, maxfz;

    maxfx=0.0;
    maxfy=0.0;
    maxfz=0.0;

    int idx_max_vx, idx_max_vy, idx_max_vz;
    cublasStatus_t stat;


    stat = cublasIdamax(handle0, 2 * ntotal_complex, fields.d_tmparray_r[VX], 1, &idx_max_vx);
    if (stat != CUBLAS_STATUS_SUCCESS) std::printf("vx max failed\n");
    stat = cublasIdamax(handle0, 2 * ntotal_complex, fields.d_tmparray_r[VY], 1, &idx_max_vy);
    if (stat != CUBLAS_STATUS_SUCCESS) std::printf("vy max failed\n");
    stat = cublasIdamax(handle0, 2 * ntotal_complex, fields.d_tmparray_r[VZ], 1, &idx_max_vz);
    if (stat != CUBLAS_STATUS_SUCCESS) std::printf("vz max failed\n");


    // index is in fortran convention
    CUDA_RT_CALL(cudaMemcpy(&maxfx, &fields.d_tmparray_r[VX][idx_max_vx-1], sizeof(scalar_type), cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaMemcpy(&maxfy, &fields.d_tmparray_r[VY][idx_max_vy-1], sizeof(scalar_type), cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaMemcpy(&maxfz, &fields.d_tmparray_r[VZ][idx_max_vz-1], sizeof(scalar_type), cudaMemcpyDeviceToHost));


    maxfx=fabs(maxfx);
    maxfy=fabs(maxfy);
    maxfz=fabs(maxfz);



    gamma_v = ( fields.wavevector.kxmax ) * maxfx + fields.wavevector.kymax * maxfy + fields.wavevector.kzmax * maxfz;


#ifdef WITH_ROTATION
    gamma_v += fabs(param.omega) / param.safety_source;
#endif

#ifdef WITH_SHEAR
    gamma_v += fabs(param.shear) / param.safety_source;
#endif

// #ifdef INCOMPRESSIBLE
#ifdef WITH_EXPLICIT_DISSIPATION
	gamma_v += ((fields.wavevector.kxmax )*( fields.wavevector.kxmax )+fields.wavevector.kymax*fields.wavevector.kymax+fields.wavevector.kzmax*fields.wavevector.kzmax) * param.nu;	// CFL condition on viscosity in incompressible regime
#endif
// #endif

#ifdef BOUSSINESQ
    gamma_v += pow(fabs(param.N2), 0.5) / param.safety_source;
#ifdef WITH_EXPLICIT_DISSIPATION
#ifndef ANISOTROPIC_DIFFUSION
    gamma_v += ((fields.wavevector.kxmax )*( fields.wavevector.kxmax )+fields.wavevector.kymax*fields.wavevector.kymax+fields.wavevector.kzmax*fields.wavevector.kzmax) * param.nu_th; // NB: this is very conservative. It should be combined with the condition on nu
#else
    gamma_v += ((fields.wavevector.kxmax )*( fields.wavevector.kxmax )+fields.wavevector.kymax*fields.wavevector.kymax+fields.wavevector.kzmax*fields.wavevector.kzmax) * (1./param.reynolds_ani); //
#endif // ANISOTROPIC_DIFFUSION
#endif // WITH_EXPLICIT_DISSIPATION
#endif // BOUSSINESQ

#ifdef DDEBUG
    if (current_step == 1 || current_step % 100 == 0 ) std::printf("maxfx: %.4e \t maxfy: %.4e \t maxfz: %.4e \t gamma_v: %.4e \n",maxfx,maxfy,maxfz,gamma_v);
#endif

#ifdef MHD
    double gamma_b;
    double maxbx, maxby, maxbz;


    maxbx=0.0;
    maxby=0.0;
    maxbz=0.0;

    int idx_max_bx, idx_max_by, idx_max_bz;
    // cublasStatus_t stat;

    // here we need to do a c2r transform first when in production

    // c2r_fft(d_farray[BX], d_farray_r[BX]);
    // c2r_fft(d_farray[BY], d_farray_r[BY]);
    // c2r_fft(d_farray[BZ], d_farray_r[BZ]);

    stat = cublasIdamax(handle0, 2 * ntotal_complex, fields.d_tmparray_r[BX], 1, &idx_max_bx);
    stat = cublasIdamax(handle0, 2 * ntotal_complex, fields.d_tmparray_r[BY], 1, &idx_max_by);
    stat = cublasIdamax(handle0, 2 * ntotal_complex, fields.d_tmparray_r[BZ], 1, &idx_max_bz);

    CUDA_RT_CALL(cudaMemcpy(&maxbx, &fields.d_tmparray_r[BX][idx_max_bx-1], sizeof(scalar_type), cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaMemcpy(&maxby, &fields.d_tmparray_r[BY][idx_max_by-1], sizeof(scalar_type), cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaMemcpy(&maxbz, &fields.d_tmparray_r[BZ][idx_max_bz-1], sizeof(scalar_type), cudaMemcpyDeviceToHost));
    // maxfx=d_farray_r[0][idx_max_vx-1];
    // maxfy=d_farray_r[1][idx_max_vy-1];
    // maxfz=d_farray_r[2][idx_max_vz-1];

    maxbx=fabs(maxbx);
    maxby=fabs(maxby);
    maxbz=fabs(maxbz);

    // std::printf("maxbx: %.5f \n",maxfx);
    // std::printf("maxby: %.5f \n",maxfy);
    // std::printf("maxbz: %.5f \n",maxfz);


    gamma_b = ( fields.wavevector.kxmax ) * maxbx + fields.wavevector.kymax * maxby + fields.wavevector.kzmax * maxbz;


#ifdef WITH_EXPLICIT_DISSIPATION
    gamma_b += ((fields.wavevector.kxmax )*( fields.wavevector.kxmax )+fields.wavevector.kymax*fields.wavevector.kymax+fields.wavevector.kzmax*fields.wavevector.kzmax) * param.nu_m;	// CFL condition on resistivity
#endif

#ifdef DDEBUG
    if (current_step == 1 || current_step % 100 == 0 ) std::printf("maxbx: %.4e \t maxby: %.4e \t maxbz: %.4e \t gamma_b: %.4e \n",maxbx,maxby,maxbz,gamma_b);
#endif

    dt = param.cfl / (gamma_v  + gamma_b);

    // r2c_fft(d_farray_r[BX], d_farray[BX]);
    // r2c_fft(d_farray_r[BY], d_farray[VY]);
    // r2c_fft(d_farray_r[BZ], d_farray[BZ]);

#else //not MHD

    dt = param.cfl / (gamma_v );

#endif //end MHD

    // r2c_fft(d_farray_r[VX], d_farray[VX]);
    // r2c_fft(d_farray_r[VY], d_farray[VY]);
    // r2c_fft(d_farray_r[VZ], d_farray[VZ]);

#endif //end INCOMPRESSIBLE

#ifdef HEAT_EQ
    double gamma_v = ((fields.wavevector.kxmax )*( fields.wavevector.kxmax )+fields.wavevector.kymax*fields.wavevector.kymax+fields.wavevector.kzmax*fields.wavevector.kzmax) * param.nu_th;

    dt = param.cfl / (gamma_v );
#endif

#ifdef DDEBUG
    if (current_step == 1 || current_step % 100 == 0 ) std::printf("t: %.4e \t dt: %.4e \n", current_time, dt);
#endif

    current_dt = dt;
    // *p_dt = dt;
}







    // absolute3<scalar_type>        unary_op;
    // thrust::maximum<scalar_type> binary_op;
    // maxfx = thrust::reduce(thrust::device_pointer_cast(d_farray_r[0]), thrust::device_pointer_cast(d_farray_r[0]) + 2*ntotal_complex, (double) 0, thrust::maximum<double>());
    // maxfx = thrust::transform_reduce(thrust::device_pointer_cast(d_farray_r[0]), thrust::device_pointer_cast(d_farray_r[0]) + 2*ntotal_complex, unary_op, (double) 0, binary_op);

    // old code that zips vx vy vz into a vector
    // Tuple3 temp;
    // auto begin = thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(d_farray_r[0]),thrust::device_pointer_cast(d_farray_r[1]), thrust::device_pointer_cast(d_farray_r[2])));
    // auto end = thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(d_farray_r[0]) + 2*ntotal_complex,thrust::device_pointer_cast(d_farray_r[1]) + 2*ntotal_complex, thrust::device_pointer_cast(d_farray_r[2]) + 2*ntotal_complex));
    // then finds 3-tuple with max vals
    // temp = thrust::transform_reduce(begin, end, absolute3<scalar_type>(), thrust::make_tuple<scalar_type,scalar_type,scalar_type>(0,0,0), MaxAbs<scalar_type>());
    // retrieve values
    // maxfx=thrust::get<0>(temp);
    // maxfy=thrust::get<1>(temp);
    // maxfz=thrust::get<2>(temp);


    // for timing
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // float milliseconds;
    //
    // int Niter = 100;
    // cudaEventRecord(start);
    // for (int ii = 0; ii < Niter; ii++) {
    //     temp = thrust::transform_reduce(begin, end, absolute3<scalar_type>(), thrust::make_tuple<scalar_type,scalar_type,scalar_type>(0,0,0), MaxAbs<scalar_type>());
    // }
    // cudaEventRecord(stop);
    // cudaDeviceSynchronize();
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::printf("THRUST elapsed time (in s): %.5f \t Approx time per reduce (in ms): %.5f \n",milliseconds/1000, milliseconds/Niter);


    // equivalent code with CUBLAS (similar speed)
    // std::printf("now with CUBLAS \n");
    // int idx_max_vx, idx_max_vy, idx_max_vz;
    // cublasStatus_t stat;
    //
    // // int Niter = 100;
    // cudaEventRecord(start);
    // for (int ii = 0; ii < Niter; ii++) {
    //     stat = cublasIdamax(handle0, 2 * ntotal_complex, d_farray_r[0], 1, &idx_max_vx);
    //     stat = cublasIdamax(handle0, 2 * ntotal_complex, d_farray_r[1], 1, &idx_max_vy);
    //     stat = cublasIdamax(handle0, 2 * ntotal_complex, d_farray_r[2], 1, &idx_max_vz);
    // }
    // cudaEventRecord(stop);
    // cudaDeviceSynchronize();
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::printf("CUBLAS elapsed time (in s): %.5f \t Approx time per reduce (in ms): %.5f \n",milliseconds/1000, milliseconds/Niter);
    // std::printf("idx_max_vx: %d \n",idx_max_vx);
    // std::printf("idx_max_vy: %d \n",idx_max_vy);
    // std::printf("idx_max_vz: %d \n",idx_max_vz);
    //
    // maxfx=d_farray_r[0][idx_max_vx];
    // maxfy=d_farray_r[1][idx_max_vy];
    // maxfz=d_farray_r[2][idx_max_vz];
