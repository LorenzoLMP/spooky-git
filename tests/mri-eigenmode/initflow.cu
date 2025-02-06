#include "common.hpp"
#include "cufft_routines.hpp"
// #include "cuda_kernels.hpp"
#include "fields.hpp"
#include "parameters.hpp"
#include "supervisor.hpp"

void Fields::initSpatialStructure(){

	int i,j,k;
	std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

	/*******************************************************************
	** This part does not need to be modified **************************
	********************************************************************/
	// Allocate coordinate arrays
	scalar_type *x, *y, *z;
	// cpudata_t x((size_t) grid.NTOTAL_COMPLEX);
    // cpudata_t y((size_t) grid.NTOTAL_COMPLEX);
    // cpudata_t z((size_t) grid.NTOTAL_COMPLEX);

	x = (scalar_type *) malloc( (size_t) sizeof(data_type) * grid.NTOTAL_COMPLEX);
	y = (scalar_type *) malloc( (size_t) sizeof(data_type) * grid.NTOTAL_COMPLEX);
	z = (scalar_type *) malloc( (size_t) sizeof(data_type) * grid.NTOTAL_COMPLEX);
 //
 //    // Initialize the arrays
	// // MPI_Printf("grid.NZ = %d \n", grid.NZ);
    #ifndef WITH_2D
	for(i = 0 ; i < grid.NX ; i++) {
		for(j = 0 ; j < grid.NY ; j++) {
			for(k = 0 ; k < grid.NZ ; k++) {
				x[k + (grid.NZ + 2) * j + (grid.NZ + 2) * grid.NY * i] = - param_ptr->lx / 2 + (param_ptr->lx * i) / grid.NX;
				y[k + (grid.NZ + 2) * j + (grid.NZ + 2) * grid.NY * i] = - param_ptr->ly / 2 + (param_ptr->ly * j ) / grid.NY;
				z[k + (grid.NZ + 2) * j + (grid.NZ + 2) * grid.NY * i] = - param_ptr->lz / 2 + (param_ptr->lz * k ) / grid.NZ;
			}
		}
		// std::printf("x[%d] = %.2e \t",i,x[(grid.NZ + 2) * grid.NY * i]);
	}
	std::cout << std::endl;
	// std::printf("coords initialized\n");
	#else
	for(i = 0 ; i < grid.NX ; i++) {
		for(j = 0 ; j < grid.NY ; j++) {
			for(k = 0 ; k < grid.NZ ; k++) {
				x[k + (grid.NZ) * j + (grid.NZ) * (grid.NY + 2) * i] = - param_ptr->lx / 2 + (param_ptr->lx * i) / grid.NX;
				y[k + (grid.NZ) * j + (grid.NZ) * (grid.NY + 2) * i] = - param_ptr->ly / 2 + (param_ptr->ly * j ) / grid.NY;
				z[k + (grid.NZ) * j + (grid.NZ) * (grid.NY + 2) * i] = - param_ptr->lz / 2 + (param_ptr->lz * k ) / grid.NZ;
			}
		}
	}
    #endif
	// Initialize the extra points (k=grid.NZ and k=grid.NZ+1) to zero to prevent stupid things from happening...
	#ifndef WITH_2D
	for(i = 0 ; i < grid.NX ; i++) {
		for(j = 0 ; j < grid.NY ; j++) {
			for(k = grid.NZ ; k < grid.NZ + 2 ; k++) {
				x[k + (grid.NZ + 2) * j + (grid.NZ + 2) * grid.NY * i] = 0.0;
				y[k + (grid.NZ + 2) * j + (grid.NZ + 2) * grid.NY * i] = 0.0;
				z[k + (grid.NZ + 2) * j + (grid.NZ + 2) * grid.NY * i] = 0.0;
			}
		}
	}
	#else
	for(i = 0 ; i < grid.NX ; i++) {
		for(j = grid.NY ; j < grid.NY + 2; j++) {
			for(k = 0 ; k < grid.NZ ; k++) {
				x[k + (grid.NZ ) * j + (grid.NZ ) * (grid.NY + 2) * i] = 0.0;
				y[k + (grid.NZ ) * j + (grid.NZ ) * (grid.NY + 2) * i] = 0.0;
				z[k + (grid.NZ ) * j + (grid.NZ ) * (grid.NY + 2) * i] = 0.0;
			}
		}
	}
	#endif


	///////////////////////////////////////
	// initial conditions on host data
	///////////////////////////////////////

	// Axisymmetric MRI eigenmode with k_x, k_z, Pm=1
	// and vertical background magnetic field

	double B0z = 0.1;
	double kZ     = 2.0*2.0*M_PI/param_ptr->lz;
	double kX     = 1.0*2.0*M_PI/param_ptr->lx;
	double gamma2 = kZ*kZ/(kZ*kZ + kX*kX);
	double kappa2 = 2*param_ptr->omega*(2*param_ptr->omega - param_ptr->shear);
	double omegaA2    = B0z*B0z*kZ*kZ;
	double Omega = param_ptr->omega;
	double S = param_ptr->shear;
	double Omega2 = Omega*Omega;

	// solve for omega_nu2, with omega_nu = -i * sigma_nu
	double omega_nu2 = omegaA2 + 0.5*kappa2*gamma2*( 1. - sqrt(1. + 16.*omegaA2*Omega2/(kappa2*kappa2*gamma2) ) );
	double sigma_nu = sqrt(-omega_nu2);
	double sigma = sigma_nu - (kX*kX + kZ*kZ)/param_ptr->reynolds;
	// double sigma_nu   = sigma + (kX*kX + kZ*kZ)/param_ptr->reynolds;
	// we assume sigma_nu and sigma_eta are equal
	double sigma_eta  = sigma + (kX*kX + kZ*kZ)/param_ptr->reynolds_m;

	double a = 0.0001;


	for (int i = 0; i < 2*grid.NTOTAL_COMPLEX; i++){

		farray_r[vars.VX][i] = a*cos(kX*x[i] + kZ*z[i]) ;
		farray_r[vars.VY][i] = farray_r[vars.VX][i] * (S *omegaA2/(sigma_eta*sigma_eta) - (2*Omega - S))/(sigma_nu + omegaA2/sigma_eta) ;
		farray_r[vars.VZ][i] = -(kX/kZ)*a*cos(kX*x[i] + kZ*z[i]);


		farray_r[vars.BX][i] = -a*sin(kX*x[i] + kZ*z[i])*B0z*kZ/sigma_eta;
		farray_r[vars.BY][i] = farray_r[vars.BX][i] * ( -S/sigma_eta + (S*omegaA2/(sigma_eta*sigma_eta) - (2*Omega - S)) /(sigma_nu + omegaA2/sigma_eta) );
		farray_r[vars.BZ][i] = B0z * (1.0 + (kX/kZ)*a*sin(kX*x[i] + kZ*z[i])*kZ/sigma_eta );


	}



	free(x);
	free(y);
	free(z);

	std::printf("Finished initializing spatial structure\n");
}
