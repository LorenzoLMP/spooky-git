#include "common.hpp"
#include "timestepping.hpp"
#include "fields.hpp"
#include "cufft_routines.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
#include "cublas_routines.hpp"
#include "parameters.hpp"
#include "physics.hpp"
#include "supervisor.hpp"
#include "inputoutput.hpp"


void TimeStepping::compute_dt(data_type* complex_Fields, scalar_type* real_Buffer) {

    NVTX3_FUNC_RANGE();

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;
    std::shared_ptr<InputOutput> inout_ptr = supervisor_ptr->inout_ptr;

    dt_par = 0.0;
    dt_hyp = 0.0;
    // double dt_tot = 0.0;
    double gamma_v = 0.0, gamma_th = 0.0, gamma_par = 0.0, gamma_b = 0.0;
    double kxmax = fields_ptr->wavevector.kxmax;
    double kymax = fields_ptr->wavevector.kymax;
    double kzmax = fields_ptr->wavevector.kzmax;

    if (param_ptr->debug > 0) {
        std::printf("Now entering compute_dt function \n");
    }

    if (param_ptr->heat_equation) {
        // for heat eq we do not need to transform from complex
        // to real, because we can just use complex variables
        // and the dt is fixed (given by nu_th)

        gamma_par = ((kxmax )*( kxmax )+kymax*kymax+kzmax*kzmax) * param_ptr->nu_th;
        dt_par = param_ptr->cfl_par / gamma_par;
        current_dt = dt_par;

        // #if defined(SUPERTIMESTEPPING) && defined(TEST)
        //     // replicate Vaidya 2017
        //     dt_hyp = 0.00703125 * (param_ptr->lx/grid.NX);
        //     current_dt = dt_hyp;
        // #endif
    }

    if (param_ptr->incompressible) {

        // for incompressible we need to first transform from
        // complex to real in order to compute dt

        // this functions copies the complex fields from d_all_fields into d_all_buffer_r and performs
        // an in-place r2c FFT to give the real fields. This buffer is reserved for the real fields!
        supervisor_ptr->Complex2RealFields(complex_Fields, real_Buffer, vars.NUM_FIELDS);

        // now we have all the real fields
        scalar_type* vx = real_Buffer + 2 * grid.NTOTAL_COMPLEX * vars.VX;
        scalar_type* vy = real_Buffer + 2 * grid.NTOTAL_COMPLEX * vars.VY;
        scalar_type* vz = real_Buffer + 2 * grid.NTOTAL_COMPLEX * vars.VZ;


        double maxfx, maxfy, maxfz;

        maxfx=0.0;
        maxfy=0.0;
        maxfz=0.0;

        int idx_max_vx, idx_max_vy, idx_max_vz;
        cublasStatus_t stat;


        stat = cublasIdamax(handle0, 2 * grid.NTOTAL_COMPLEX, vx, 1, &idx_max_vx);
        if (stat != CUBLAS_STATUS_SUCCESS) std::printf("vx max failed\n");
        stat = cublasIdamax(handle0, 2 * grid.NTOTAL_COMPLEX, vy, 1, &idx_max_vy);
        if (stat != CUBLAS_STATUS_SUCCESS) std::printf("vy max failed\n");
        stat = cublasIdamax(handle0, 2 * grid.NTOTAL_COMPLEX, vz, 1, &idx_max_vz);
        if (stat != CUBLAS_STATUS_SUCCESS) std::printf("vz max failed\n");


        // index is in fortran convention
        CUDA_RT_CALL(cudaMemcpy(&maxfx, &vx[idx_max_vx-1], sizeof(scalar_type), cudaMemcpyDeviceToHost));
        CUDA_RT_CALL(cudaMemcpy(&maxfy, &vy[idx_max_vy-1], sizeof(scalar_type), cudaMemcpyDeviceToHost));
        CUDA_RT_CALL(cudaMemcpy(&maxfz, &vz[idx_max_vz-1], sizeof(scalar_type), cudaMemcpyDeviceToHost));


        maxfx=fabs(maxfx);
        maxfy=fabs(maxfy);
        maxfz=fabs(maxfz);



        gamma_v = ( kxmax + fabs(tremap)*kymax ) * maxfx + kymax * maxfy + kzmax * maxfz;

        if (param_ptr->rotating) {
            gamma_v += fabs(param_ptr->omega) / param_ptr->safety_source;
        }

        if (param_ptr->shearing) {
            gamma_v += fabs(param_ptr->shear) / param_ptr->safety_source;
        }

        gamma_par += ((kxmax + fabs(tremap)*kymax )*( kxmax + fabs(tremap)*kymax)+kymax*kymax+kzmax*kzmax) * param_ptr->nu;	// CFL condition on viscosity in incompressible regime


        if (param_ptr->boussinesq) {
            gamma_th += pow(fabs(param_ptr->N2), 0.5) / param_ptr->safety_source;

            if (param_ptr->anisotropic_diffusion) {

                gamma_th += pow(fabs(param_ptr->OmegaT2), 0.5) / param_ptr->safety_source;

                gamma_par += ((kxmax + fabs(tremap)*kymax )*( kxmax + fabs(tremap)*kymax)+kymax*kymax+kzmax*kzmax) * (1./param_ptr->reynolds_ani);
            }
            else {
                gamma_par += ((kxmax + fabs(tremap)*kymax)*( kxmax + fabs(tremap)*kymax)+kymax*kymax+kzmax*kzmax) * param_ptr->nu_th; // NB: this is very conservative. It should be combined with the condition on nu
            }
        }

        if (param_ptr->debug > 1) {
            std::printf("maxfx: %.6e \t maxfy: %.6e \t maxfz: %.6e \t gamma_v: %.6e \n",maxfx,maxfy,maxfz,gamma_v);
        }

        if (param_ptr->mhd) {

            double maxbx, maxby, maxbz;

            maxbx=0.0;
            maxby=0.0;
            maxbz=0.0;

            int idx_max_bx, idx_max_by, idx_max_bz;
            // cublasStatus_t stat;

            scalar_type* Bx = real_Buffer + 2 * grid.NTOTAL_COMPLEX * vars.BX;
            scalar_type* By = real_Buffer + 2 * grid.NTOTAL_COMPLEX * vars.BY;
            scalar_type* Bz = real_Buffer + 2 * grid.NTOTAL_COMPLEX * vars.BZ;

            stat = cublasIdamax(handle0, 2 * grid.NTOTAL_COMPLEX, Bx, 1, &idx_max_bx);
            stat = cublasIdamax(handle0, 2 * grid.NTOTAL_COMPLEX, By, 1, &idx_max_by);
            stat = cublasIdamax(handle0, 2 * grid.NTOTAL_COMPLEX, Bz, 1, &idx_max_bz);

            CUDA_RT_CALL(cudaMemcpy(&maxbx, &Bx[idx_max_bx-1], sizeof(scalar_type), cudaMemcpyDeviceToHost));
            CUDA_RT_CALL(cudaMemcpy(&maxby, &By[idx_max_by-1], sizeof(scalar_type), cudaMemcpyDeviceToHost));
            CUDA_RT_CALL(cudaMemcpy(&maxbz, &Bz[idx_max_bz-1], sizeof(scalar_type), cudaMemcpyDeviceToHost));

            maxbx=fabs(maxbx);
            maxby=fabs(maxby);
            maxbz=fabs(maxbz);

            gamma_b = ( kxmax + fabs(tremap)*kymax) * maxbx + kymax * maxby + kzmax * maxbz;

            gamma_par += ((kxmax + fabs(tremap)*kymax)*( kxmax + fabs(tremap)*kymax)+kymax*kymax+kzmax*kzmax) * param_ptr->nu_m;	// CFL condition on resistivity

            if (param_ptr->debug > 1) {
                std::printf("maxbx: %.6e \t maxby: %.6e \t maxbz: %.6e \t gamma_b: %.6e \n",maxbx,maxby,maxbz,gamma_b);
            }

        } //end MHD

        dt_hyp = param_ptr->cfl / (gamma_v + gamma_th + gamma_b);
        dt_par = param_ptr->cfl_par / gamma_par;
        // dt_tot = param_ptr->cfl / (gamma_v + gamma_th + gamma_b + gamma_par);
        // dt_tot = 1.0 / (1.0/dt_hyp + 1.0/dt_par);

        if (not param_ptr->supertimestepping) {

            current_dt = param_ptr->cfl / (gamma_v + gamma_th + gamma_b + gamma_par);
        }
        else {

            if ( dt_hyp > dt_par * param_ptr->safety_sts) {
                dt_hyp =  dt_par * param_ptr->safety_sts;
            }
            // the following is checked later
            // if ( dt_hyp < dt_par ) {
            //     dt_par = dt_hyp;
            // }
            current_dt = dt_hyp;
        }

    } //end INCOMPRESSIBLE


    // this is to stop exactly at t_final
    // or at the t_output flow
    if ( current_time + current_dt > param_ptr->t_final) {
        current_dt = param_ptr->t_final - current_time;
    }
    else if ( current_time + current_dt - inout_ptr->t_lastsnap > param_ptr->toutput_flow) {
        current_dt = param_ptr->toutput_flow - current_time + inout_ptr->t_lastsnap;
    }
    // when using sts dt_par may also have to
    // be shrunk accordingly
    if (param_ptr->supertimestepping) {
        if ( current_dt < dt_par ) {
            dt_par = current_dt;
        }
    }


    if (param_ptr->debug > 0) {

        std::printf("t: %.4e \t gamma_par = %.4e \t gamma_v = %.4e \t gamma_b = %.4e \t dt_hyp: %.4e \t dt_par: %.4e \t dt_current: %.4e \n", current_time, gamma_par, gamma_v + gamma_th, gamma_b, dt_hyp, dt_par, current_dt);
    }

}

