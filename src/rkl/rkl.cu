#include "define_types.hpp"
#include "rkl.hpp"
// #include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "cublas_routines.hpp"
#include "cuda_kernels.hpp"
#include "parameters.hpp"
// #include "inputoutput.hpp"
#include "fields.hpp"
#include "physics.hpp"
#include <cuda_runtime.h>
// #include <cufftXt.h>
// #include "spooky.hpp"
#include "cufft_utils.h"
// #include "define_types.hpp"
#include "supervisor.hpp"
#include "timestepping.hpp"

RKLegendre::RKLegendre(int num, Parameters &param, Supervisor &sup) : ts(STS_MAX_STEPS, 0.0) {
    // param = &p_in;
    // fields = &f_in;

    supervisor = &sup;
    // std::printf("The TimeSpentInFFTs is: %.4e",supervisor->TimeSpentInFFTs);
    dt = 0.0;
    stage = 0;
    cfl_rkl = param.cfl_par;
    rmax_par = param.safety_sts;
    // std::vector<double> ts(STS_MAX_STEPS, 0.0);

    blocksPerGrid = ( 2 * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    // this is the mega array that contains intermediate fields during multi-stage timestepping
    // std::printf("num fields ts: %d \n", fields->num_fields);
    std::printf("num rkl scratch arrays: %d \n",4);


    CUDA_RT_CALL(cudaMalloc(&d_all_dU, (size_t) sizeof(data_type) * ntotal_complex));
    CUDA_RT_CALL(cudaMalloc(&d_all_dU0, (size_t) sizeof(data_type) * ntotal_complex));
    CUDA_RT_CALL(cudaMalloc(&d_all_Uc0, (size_t) sizeof(data_type) * ntotal_complex));
    CUDA_RT_CALL(cudaMalloc(&d_all_Uc1, (size_t) sizeof(data_type) * ntotal_complex));
}

RKLegendre::~RKLegendre(){
    CUDA_RT_CALL(cudaFree(d_all_dU));
    CUDA_RT_CALL(cudaFree(d_all_dU0));
    CUDA_RT_CALL(cudaFree(d_all_Uc0));
    CUDA_RT_CALL(cudaFree(d_all_Uc1));
}


double STS_CorrectTimeStep(int n0, double dta, double STS_NU);
double STS_FindRoot(double dt_exp, double dT, double STS_NU);
void STS_ComputeSubSteps(double dtex, double tau[], int N, double STS_NU);


void RKLegendre::compute_cycle_STS(Fields &fields, TimeStepping &timestep, Physics &phys){

    double dt_hyp = timestep.current_dt;
    double dt_par_corr = timestep.dt_par;


    int i;
    int nv, n, m;
    double N;
    int nv_indx, nvar_rkl;
    double tau;

#ifdef BOUSSINESQ
#ifdef SUPERTIMESTEPPING

    tau = dt_par_corr;

    m = 0;
    n = STS_MAX_STEPS;
    while (m < n){

        N = STS_FindRoot(dt_par, dt_hyp, STS_NU);
        N = floor(N+1.0);
        n = (int)N;

        if (n > 1){
            dt_par_corr = STS_CorrectTimeStep(n, dt_hyp, STS_NU);
            STS_ComputeSubSteps(dt_par_corr, ts, n, STS_NU);
        }
        if (n == 1) ts[0] = dt_hyp;
        tau = ts[n-m-1];


        // anisotropic_conduction( rhs, fldi);
        AnisotropicConduction(fields, param, (data_type *) fields.d_farray[TH], (data_type *) d_all_dU);

        // update fields.d_farray[TH]
        axpyDouble<<<blocksPerGrid, threadsPerBlock>>>((scalar_type *) fields.d_farray[TH], (scalar_type *) d_all_dU, (scalar_type *) fields.d_farray[TH], 1.0, tau,  2 * ntotal_complex);
        // for( i = 0 ; i < NTOTAL_COMPLEX ; i++){
        //     fldi.farray[nv_indx][i] += tau*rhs.farray[nv_indx][i];
        //     rhs.farray[nv_indx][i] = 0.0;
        // }
        m++;
    }


#endif //supertimestepping
#endif // Boussinesq
}


void STS_ComputeSubSteps(double dtex, double tau[], int N, double STS_NU)
/*!
 * Compute the single sub-step sequence (Eq. [2.9]).
 * N must be an integer by now.
 *
 *********************************************************************** */
{
  int i;
  double S=0.0;

  for (i = 0; i < N; i++) {
    tau[i] = dtex / ((-1.0 + STS_NU)*cos(((2.0*i+1.0)*M_PI)/(2.0*N))
                     + 1.0 + STS_NU);
    S += tau[i];
  }
}

/* ********************************************************************* */
double STS_FindRoot(double dt_exp, double dT, double STS_NU)
/*!
 * Find the number of sub-steps N by solving Eq. (2.10) of AAG using a
 * Newton-Raphson scheme.
 * Input to the function are:
 *
 * \param [in]  dt_exp   the explicit time step
 * \param [in]  dt       the super-step.
 *
 *********************************************************************** */
{
  int k;  /* Iteration number */
  double a,b,c, scrh;
  double fN, N, dN, dfN;
  double db_dN, sqrt_nu = sqrt(STS_NU);

  k = 0;
  N = 1.0;
  a = (1.0 - sqrt_nu)/(1.0 + sqrt_nu);
  while(k < 128){
    b     = pow(a,2.0*N);
    c     = (1.0-b)/(1.0+b);    /* round bracket in Eq. [10] in AAG */
    db_dN = 2.0*log(a)*b;
    scrh  = c - N*2.0/((1.0+b)*(1.0+b))*db_dN;

    fN  = dT - 0.5*dt_exp/sqrt_nu*N*c;
    dfN =    - 0.5*dt_exp/sqrt_nu*scrh;
    dN  = fN/dfN;

    N -= dN;
    k++;

    if (fabs(dN) < 1.e-5) return N;
  }
  return -1.0;
}

/* ********************************************************************* */
double STS_CorrectTimeStep(int n0, double dta, double STS_NU)
/*
 *
 *********************************************************************** */
{
  double a,b,c;
  double dtr;

  a = (1.0-sqrt(STS_NU))/(1.0+sqrt(STS_NU));
  b = pow(a,2.0*n0);
  c = (1.0-b)/(1.0+b);

  dtr = dta*2.0*sqrt(STS_NU)/(n0*c);
  return(dtr);
}
