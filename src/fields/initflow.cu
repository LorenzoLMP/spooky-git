#include "define_types.hpp"
#include "cufft_routines.hpp"
// #include "cuda_kernels.hpp"
#include "fields.hpp"
#include "spooky.hpp"
#include "common.hpp"
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
	// cpudata_t x((size_t) ntotal_complex);
    // cpudata_t y((size_t) ntotal_complex);
    // cpudata_t z((size_t) ntotal_complex);

	x = (scalar_type *) malloc( (size_t) sizeof(data_type) * ntotal_complex);
	y = (scalar_type *) malloc( (size_t) sizeof(data_type) * ntotal_complex);
	z = (scalar_type *) malloc( (size_t) sizeof(data_type) * ntotal_complex);
 //
 //    // Initialize the arrays
	// // MPI_Printf("nz = %d \n", nz);
    #ifndef WITH_2D
	for(i = 0 ; i < nx ; i++) {
		for(j = 0 ; j < ny ; j++) {
			for(k = 0 ; k < nz ; k++) {
				x[k + (nz + 2) * j + (nz + 2) * ny * i] = - param_ptr->lx / 2 + (param_ptr->lx * i) / nx;
				y[k + (nz + 2) * j + (nz + 2) * ny * i] = - param_ptr->ly / 2 + (param_ptr->ly * j ) / ny;
				z[k + (nz + 2) * j + (nz + 2) * ny * i] = - param_ptr->lz / 2 + (param_ptr->lz * k ) / nz;
			}
		}
		// std::printf("x[%d] = %.2e \t",i,x[(nz + 2) * ny * i]);
	}
	std::cout << std::endl;
	// std::printf("coords initialized\n");
	#else
	for(i = 0 ; i < nx ; i++) {
		for(j = 0 ; j < ny ; j++) {
			for(k = 0 ; k < nz ; k++) {
				x[k + (nz) * j + (nz) * (ny + 2) * i] = - param_ptr->lx / 2 + (param_ptr->lx * i) / nx;
				y[k + (nz) * j + (nz) * (ny + 2) * i] = - param_ptr->ly / 2 + (param_ptr->ly * j ) / ny;
				z[k + (nz) * j + (nz) * (ny + 2) * i] = - param_ptr->lz / 2 + (param_ptr->lz * k ) / nz;
			}
		}
	}
    #endif
	// Initialize the extra points (k=nz and k=nz+1) to zero to prevent stupid things from happening...
	#ifndef WITH_2D
	for(i = 0 ; i < nx ; i++) {
		for(j = 0 ; j < ny ; j++) {
			for(k = nz ; k < nz + 2 ; k++) {
				x[k + (nz + 2) * j + (nz + 2) * ny * i] = 0.0;
				y[k + (nz + 2) * j + (nz + 2) * ny * i] = 0.0;
				z[k + (nz + 2) * j + (nz + 2) * ny * i] = 0.0;
			}
		}
	}
	#else
	for(i = 0 ; i < nx ; i++) {
		for(j = ny ; j < ny + 2; j++) {
			for(k = 0 ; k < nz ; k++) {
				x[k + (nz ) * j + (nz ) * (ny + 2) * i] = 0.0;
				y[k + (nz ) * j + (nz ) * (ny + 2) * i] = 0.0;
				z[k + (nz ) * j + (nz ) * (ny + 2) * i] = 0.0;
			}
		}
	}
	#endif


	///////////////////////////////////////
	// initial conditions on host data
	///////////////////////////////////////
	double a = 0.01;
	for (int i = 0; i < 2*ntotal_complex; i++){
		if (param_ptr->heat_equation){
			farray_r[TH][i] = 1.0 +  0.5 * (tanh((x[i] + 0.375) / a) - tanh((x[i] + 0.125) / a)) + 0.5 * (tanh((x[i] - 0.125) / a) - tanh((x[i] - 0.375) / a));
		}
		if (param_ptr->incompressible){
			farray_r[VX][i] =   sin(2.0*M_PI*x[i]/param_ptr->lx) * cos(2.0*M_PI*y[i]/param_ptr->ly);
			farray_r[VY][i] = - cos(2.0*M_PI*x[i]/param_ptr->lx) * sin(2.0*M_PI*y[i]/param_ptr->ly);
			farray_r[VZ][i] = 0.0;
		}
		if (param_ptr->mhd){
			farray_r[BX][i] =   sin(2.0*M_PI*x[i]/param_ptr->lx) * cos(2.0*M_PI*y[i]/param_ptr->ly);
			farray_r[BY][i] = - cos(2.0*M_PI*x[i]/param_ptr->lx) * sin(2.0*M_PI*y[i]/param_ptr->ly);
			farray_r[BZ][i] = 0.0;
		}
		if (param_ptr->boussinesq) {
			farray_r[TH][i] = 0.0;
		}

	}


	free(x);
	free(y);
	free(z);

	std::printf("Finished initializing spatial structure\n");
}
