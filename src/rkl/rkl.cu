#include "define_types.hpp"
#include "rkl.hpp"
// #include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "cublas_routines.hpp"
#include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
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
// #include <cstdlib>
#include <math.h>


RKLegendre::RKLegendre(int num_fields, Parameters &p_in, Supervisor &sup_in) {
    // param = &p_in;
    // fields = &f_in;

    supervisor_ptr = &sup_in;
    // ts = (double*)malloc(sizeof(double)*STS_MAX_STEPS);
    ts = new double[STS_MAX_STEPS];
    for (int i = 0; i < STS_MAX_STEPS; i++){
      ts[i] = 0.0;
    }

    // std::printf("The TimeSpentInFFTs is: %.4e",supervisor_ptr->TimeSpentInFFTs);
    dt = 0.0;
    stage = 0;
    cfl_rkl = p_in.cfl_par;
    rmax_par = p_in.safety_sts;
    // std::vector<double> ts(STS_MAX_STEPS, 0.0);

    blocksPerGrid = ( num_fields * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    // this is the mega array that contains intermediate fields during multi-stage timestepping
    // std::printf("num fields ts: %d \n", fields_ptr->num_fields);
    std::printf("num rkl scratch arrays: %d \n",4);
    std::printf("blocksPerGrid: %d \n",blocksPerGrid);


    CUDA_RT_CALL(cudaMalloc(&d_all_dU, (size_t) sizeof(data_type) * ntotal_complex * num_fields));
    CUDA_RT_CALL(cudaMalloc(&d_all_dU0, (size_t) sizeof(data_type) * ntotal_complex * num_fields));
    CUDA_RT_CALL(cudaMalloc(&d_all_Uc0, (size_t) sizeof(data_type) * ntotal_complex * num_fields));
    CUDA_RT_CALL(cudaMalloc(&d_all_Uc1, (size_t) sizeof(data_type) * ntotal_complex * num_fields));

    VecInitComplex<<<blocksPerGrid, threadsPerBlock>>>(d_all_dU,  data_type(0.0,0.0), ntotal_complex * num_fields);
    VecInitComplex<<<blocksPerGrid, threadsPerBlock>>>(d_all_dU0, data_type(0.0,0.0), ntotal_complex * num_fields);
    VecInitComplex<<<blocksPerGrid, threadsPerBlock>>>(d_all_Uc0, data_type(0.0,0.0), ntotal_complex * num_fields);
    VecInitComplex<<<blocksPerGrid, threadsPerBlock>>>(d_all_Uc1, data_type(0.0,0.0), ntotal_complex * num_fields);

    d_farray_dU  = new data_type*[num_fields];
    d_farray_dU0 = new data_type*[num_fields];
    d_farray_Uc0 = new data_type*[num_fields];
    d_farray_Uc1 = new data_type*[num_fields];

    for (int i = 0; i < num_fields; i++){
      d_farray_dU[i]   = d_all_dU + i*ntotal_complex;
      d_farray_dU0[i]   = d_all_dU0 + i*ntotal_complex;
      d_farray_Uc0[i]   = d_all_Uc0 + i*ntotal_complex;
      d_farray_Uc1[i]   = d_all_Uc1 + i*ntotal_complex;
    }

}

RKLegendre::~RKLegendre(){
    CUDA_RT_CALL(cudaFree(d_all_dU));
    CUDA_RT_CALL(cudaFree(d_all_dU0));
    CUDA_RT_CALL(cudaFree(d_all_Uc0));
    CUDA_RT_CALL(cudaFree(d_all_Uc1));

    delete ts;
    delete d_farray_dU, d_farray_dU0, d_farray_Uc0, d_farray_Uc1;
}


double STS_CorrectTimeStep(int n0, double dta, double STS_NU);
double STS_FindRoot(double dt_exp, double dT, double STS_NU);
void STS_ComputeSubSteps(double dtex, double* tau, int N, double STS_NU);


void RKLegendre::compute_cycle_STS(data_type* complex_Fields, scalar_type* real_Buffer){

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    // std::shared_ptr<Parameters> param = supervisor_ptr->param;
    std::shared_ptr<TimeStepping> timestep_ptr = supervisor_ptr->timestep_ptr;
    std::shared_ptr<Physics> phys_ptr = supervisor_ptr->phys_ptr;

    double dt_hyp = timestep_ptr->current_dt;
    double dt_par = timestep_ptr->dt_par;
    double dt_par_corr = dt_par;

    // std::printf("now in supertimestepping function");
    int i;
    int nv, n, m;
    double N;
    int nv_indx, nvar_rkl;
    double tau;
    int num_fields = fields_ptr->num_fields;

// #ifdef BOUSSINESQ
#ifdef SUPERTIMESTEPPING
// #ifdef ANISOTROPIC_DIFFUSION
    tau = dt_par;

    m = 0;
    n = STS_MAX_STEPS;
    while (m < n){

        N = STS_FindRoot(dt_par, dt_hyp, STS_NU);
        N = floor(N+1.0);
        n = (int)N;
#ifdef DEBUG
        std::printf("STS::::: number of STS subcycles: %d \n",n);
#endif


        if (n > 1){
            dt_par_corr = STS_CorrectTimeStep(n, dt_hyp, STS_NU);
#ifdef DEBUG
        std::printf("STS::::: dt_par_corr: %4.e \n",dt_par_corr);
#endif
            STS_ComputeSubSteps(dt_par_corr, ts, n, STS_NU);
        }
        if (n == 1) ts[0] = dt_hyp;
        tau = ts[n-m-1];
#ifdef DEBUG
        std::printf("STS::::: tau: %4.e \n",tau);
#endif

        // anisotropic_conduction( rhs, fldi);
        // phys_ptr->AnisotropicConduction(fields, param, (data_type *) fields_ptr->d_farray[TH], (data_type *) d_farray_dU[TH]);
        // this is for all parabolic terms
        phys_ptr->ParabolicTerms(complex_Fields, real_Buffer, d_all_dU);



        // only for temperature
        // blocksPerGrid = (ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
        // addReset<<<blocksPerGrid, threadsPerBlock>>>( fields_ptr->d_farray[TH],  d_farray_dU[TH],  fields_ptr->d_farray[TH], 1.0, tau, ntotal_complex);
        // CUDA_RT_CALL( cudaDeviceSynchronize() );
        // this is for all parabolic terms
        blocksPerGrid = (ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
        for (nv = 0; nv < num_fields; nv++){
          addReset<<<blocksPerGrid, threadsPerBlock>>>( complex_Fields + nv * ntotal_complex,  d_farray_dU[nv],  complex_Fields + nv * ntotal_complex, 1.0, tau, ntotal_complex);
        }
        CUDA_RT_CALL( cudaDeviceSynchronize() );

        m++;
    }

// #endif // ANISOTROPIC_DIFFUSION
#endif //supertimestepping
// #endif // Boussinesq
}

void RKLegendre::compute_cycle_RKL(data_type* complex_Fields, scalar_type* real_Buffer){

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    // std::shared_ptr<Parameters> param = supervisor_ptr->param;
    std::shared_ptr<TimeStepping> timestep_ptr = supervisor_ptr->timestep_ptr;
    std::shared_ptr<Physics> phys_ptr = supervisor_ptr->phys_ptr;

    double dt_hyp = timestep_ptr->current_dt;
    double dt_par = timestep_ptr->dt_par;
    double time = timestep_ptr->current_time;
    int num_fields = fields_ptr->num_fields;
    // std::printf("now in supertimestepping function");


    // tau is dt_hyp
    // static Data_Arr Y_jm1, Y_jm2, MY_jm1, MY_0;
    // in idefix they correspond to:
    // Y_jm1  --> Uc0    // field step j-1
    // Y_jm2  --> Uc1   // field step j-2
    // MY_jm1 --> dU    // dfield step j-1
    // MY_0   --> dU0   // dfield step 0
    // static double **v;
    double s_str;                          /* The "s" parameter */

    int i;
    int nv, n, m, s, s_RKL = 0;
    double N, scrh;
    int nv_indx, var_list[num_fields], nvar_rkl;
    double mu_j, nu_j, mu_tilde_j, gamma_j;
    data_type Y;
    double a_jm1, b_j, b_jm1, b_jm2, w1;

// #ifdef BOUSSINESQ
#ifdef SUPERTIMESTEPPING
// #ifdef ANISOTROPIC_DIFFUSION

    scrh  = dt_hyp/dt_par;                      /*  Solution of quadratic Eq.   */
    s_str =   4.0*(1.0 + 2.0*scrh)           /*  4*tau/dt_exp = s^2 + s - 2  */
            /(1.0 + sqrt(9.0 + 16.0*scrh));

    s_RKL = 1 + int(s_str);
#ifdef DEBUG
    std::printf("RKL::::: number of RKL subcycles: %d \n",s_RKL);
#endif
    w1 = 4.0/(s_RKL*s_RKL + s_RKL - 2.0);
    mu_tilde_j = w1/3.0;

    b_j = b_jm1 = b_jm2 = 1.0/3.0;
    a_jm1 = 1.0 - b_jm1;


    // initialize temp fields
    // MY_0 <- parabolicRHS(d_farray[TH])
    blocksPerGrid = ( num_fields * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
    VecInitComplex<<<blocksPerGrid, threadsPerBlock>>>((data_type *)d_all_dU0, data_type(0.0,0.0), num_fields * ntotal_complex);

    // this is only for temperature
    // phys_ptr->AnisotropicConduction(fields, param, (data_type *) fields_ptr->d_farray[TH], (data_type *) d_farray_dU0[TH]);

    // this is for all parabolic terms
    phys_ptr->ParabolicTerms(complex_Fields, real_Buffer, d_all_dU0);

    for (nv = 0; nv < num_fields; nv++){

      // Y_jm1 <- d_farray[TH]
      blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;

      ComplexVecAssign<<<blocksPerGrid, threadsPerBlock>>>(complex_Fields + nv * ntotal_complex, d_farray_Uc0[nv], ntotal_complex);

      // Y_jm2 <- d_farray[TH]
      ComplexVecAssign<<<blocksPerGrid, threadsPerBlock>>>(complex_Fields + nv * ntotal_complex, d_farray_Uc1[nv], ntotal_complex);

      // Y_jm1 (d_farray[TH]) <- Y_jm2 + mu_tilde_j*dt_hyp*MY_0
      axpyComplex<<<blocksPerGrid, threadsPerBlock>>>( d_farray_Uc1[nv],  d_farray_dU[nv],  complex_Fields + nv * ntotal_complex, 1.0, mu_tilde_j*dt_hyp,  ntotal_complex);
    }

    /* s loop */
    s = 1;
    // g_time = t0 + 0.25*tau*(s*s+s-2)*w1;
    for (s = 2; s <= s_RKL; s++) {

      mu_j       = (2.*s -1.)/s * b_j/b_jm1;   /* Eq. [17] */
      mu_tilde_j = w1*mu_j;
      gamma_j    = -a_jm1*mu_tilde_j;
      nu_j       = -(s -1.)*b_j/(s*b_jm2);

      b_jm2 = b_jm1;    /* Eq. [16] */
      b_jm1 = b_j;
      a_jm1 = 1.0 - b_jm1;
      b_j   = 0.5*(s*s+3.0*s)/(s*s+3.0*s+2);

      blocksPerGrid = ( num_fields * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
      VecInitComplex<<<blocksPerGrid, threadsPerBlock>>>((data_type *)d_all_dU, data_type(0.0,0.0), num_fields * ntotal_complex);

      // phys_ptr->AnisotropicConduction(fields, param, (data_type *) fields_ptr->d_farray[TH], (data_type *) d_farray_dU[TH]);

      phys_ptr->ParabolicTerms(complex_Fields, real_Buffer, d_all_dU);

      for (nv = 0; nv < num_fields; nv++){
        // MY_j-1 <- parabolicRHS(d_farray[TH])

        blocksPerGrid = ( ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
        // real Y = mu_j*Uc(nv,k,j,i) + nu_j*Uc1(nv,k,j,i);
        // Uc1(nv,k,j,i) = Uc(nv,k,j,i);
        // Uc <- Y + (1.0 - mu_j - nu_j)*Uc0 + dt_hyp*mu_tilde_j*dU +  gamma_j*dt_hyp*dU0;
        axpy5ComplexAssign<<<blocksPerGrid, threadsPerBlock>>>((data_type *) complex_Fields + nv * ntotal_complex, (data_type *) d_farray_Uc1[nv], (data_type *) d_farray_Uc0[nv], (data_type *) d_farray_dU[nv], (data_type *) d_farray_dU0[nv], mu_j, nu_j, (1.0 - mu_j - nu_j), dt_hyp*mu_tilde_j,  gamma_j*dt_hyp, ntotal_complex);

        // increment time
        time = timestep_ptr->current_time + 0.25*dt_hyp*(s*s+s-2)*w1;
      }
    }


// #endif // ANISOTROPIC_DIFFUSION
#endif //supertimestepping
// #endif // Boussinesq
}


void STS_CompuFields* fields_ptrteSubSteps(double dtex, double* tau, int N, double STS_NU)
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
