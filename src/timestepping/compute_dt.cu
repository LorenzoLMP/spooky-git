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
    dt_par = 0.0;
    dt_hyp = 0.0;
    // double dt_tot = 0.0;
    double gamma_v = 0.0, gamma_th = 0.0, gamma_par = 0.0, gamma_b = 0.0;

#ifdef DDEBUG
    std::printf("Now entering compute_dt function \n");
#endif

#ifdef INCOMPRESSIBLE

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
// #ifdef WITH_EXPLICIT_DISSIPATION
	gamma_v += ((fields.wavevector.kxmax )*( fields.wavevector.kxmax )+fields.wavevector.kymax*fields.wavevector.kymax+fields.wavevector.kzmax*fields.wavevector.kzmax) * param.nu;	// CFL condition on viscosity in incompressible regime
// #endif
// #endif

#ifdef BOUSSINESQ
    gamma_th += pow(fabs(param.N2), 0.5) / param.safety_source;
#ifdef ANISOTROPIC_DIFFUSION
    gamma_th += pow(fabs(param.OmegaT2), 0.5) / param.safety_source;

    gamma_par += ((fields.wavevector.kxmax )*( fields.wavevector.kxmax )+fields.wavevector.kymax*fields.wavevector.kymax+fields.wavevector.kzmax*fields.wavevector.kzmax) * (1./param.reynolds_ani);
#else
    gamma_par += ((fields.wavevector.kxmax )*( fields.wavevector.kxmax )+fields.wavevector.kymax*fields.wavevector.kymax+fields.wavevector.kzmax*fields.wavevector.kzmax) * param.nu_th; // NB: this is very conservative. It should be combined with the condition on nu
#endif // ANISOTROPIC_DIFFUSION
#endif // BOUSSINESQ

#ifdef DDEBUG
    // if (current_step == 1 || current_step % 10 == 0 ) std::printf("maxfx: %.4e \t maxfy: %.4e \t maxfz: %.4e \t gamma_v: %.4e \n",maxfx,maxfy,maxfz,gamma_v);
    std::printf("maxfx: %.6e \t maxfy: %.6e \t maxfz: %.6e \t gamma_v: %.6e \n",maxfx,maxfy,maxfz,gamma_v);
#endif

#ifdef MHD

    double maxbx, maxby, maxbz;


    maxbx=0.0;
    maxby=0.0;
    maxbz=0.0;

    int idx_max_bx, idx_max_by, idx_max_bz;
    // cublasStatus_t stat;


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

    gamma_b += ((fields.wavevector.kxmax )*( fields.wavevector.kxmax )+fields.wavevector.kymax*fields.wavevector.kymax+fields.wavevector.kzmax*fields.wavevector.kzmax) * param.nu_m;	// CFL condition on resistivity

#ifdef DDEBUG
    // if (current_step == 1 || current_step % 10 == 0 ) std::printf("maxbx: %.4e \t maxby: %.4e \t maxbz: %.4e \t gamma_b: %.4e \n",maxbx,maxby,maxbz,gamma_b);
    std::printf("maxbx: %.6e \t maxby: %.6e \t maxbz: %.6e \t gamma_b: %.6e \n",maxbx,maxby,maxbz,gamma_b);
#endif


#endif //end MHD

    dt_hyp = param.cfl / (gamma_v + gamma_th + gamma_b);
    dt_par = param.cfl_par / gamma_par;
    // dt_tot = param.cfl / (gamma_v + gamma_th + gamma_b + gamma_par);
    // dt_tot = 1.0 / (1.0/dt_hyp + 1.0/dt_par);

#ifndef SUPERTIMESTEPPING
    current_dt = param.cfl / (gamma_v + gamma_th + gamma_b + gamma_par);

#else
    if ( dt_hyp > dt_par * param.safety_sts) {
        dt_hyp =  dt_par * param.safety_sts;
    }
    if ( dt_hyp < dt_par ) {
        dt_par = dt_hyp;
    }
    current_dt = dt_hyp;
#endif


#endif //end INCOMPRESSIBLE

#ifdef HEAT_EQ
    gamma_v = ((fields.wavevector.kxmax )*( fields.wavevector.kxmax )+fields.wavevector.kymax*fields.wavevector.kymax+fields.wavevector.kzmax*fields.wavevector.kzmax) * param.nu_th;

    current_dt = param.cfl / (gamma_v );
#endif

    if ( current_time + current_dt > param.t_final) current_dt = param.t_final - current_time;

// #ifdef DDEBUG
//     if (current_step == 1 || current_step % 100 == 0 ) std::printf("t: %.4e \t dt: %.4e \n", current_time, current_dt);
// #endif
#ifdef DEBUG
    // if (current_step == 1 || current_step % 10 == 0 ) std::printf("t: %.4e \t gamma_par = %.4e \t gamma_v = %.4e \t gamma_b = %.4e \t dt_hyp: %.4e \t dt_par: %.4e \t dt_current: %.4e \n", current_time, gamma_par, gamma_v + gamma_th, gamma_b, dt_hyp, dt_par, current_dt);
    std::printf("t: %.4e \t gamma_par = %.4e \t gamma_v = %.4e \t gamma_b = %.4e \t dt_hyp: %.4e \t dt_par: %.4e \t dt_current: %.4e \n", current_time, gamma_par, gamma_v + gamma_th, gamma_b, dt_hyp, dt_par, current_dt);
#endif

}

