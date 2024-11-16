#include "define_types.hpp"
#include "cufft_routines.hpp"
// #include "cuda_kernels.hpp"
#include "fields.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "parameters.hpp"
// #include "supervisor.hpp"

void Fields::init_SpatialStructure(Parameters &param){

	int i,j,k;
	// std::shared_ptr<Parameters> param = supervisor->param;

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
				x[k + (nz + 2) * j + (nz + 2) * ny * i] = - param.lx / 2 + (param.lx * i) / nx;
				y[k + (nz + 2) * j + (nz + 2) * ny * i] = - param.ly / 2 + (param.ly * j ) / ny;
				z[k + (nz + 2) * j + (nz + 2) * ny * i] = - param.lz / 2 + (param.lz * k ) / nz;
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
				x[k + (nz) * j + (nz) * (ny + 2) * i] = - param.lx / 2 + (param.lx * i) / nx;
				y[k + (nz) * j + (nz) * (ny + 2) * i] = - param.ly / 2 + (param.ly * j ) / ny;
				z[k + (nz) * j + (nz) * (ny + 2) * i] = - param.lz / 2 + (param.lz * k ) / nz;
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
	#ifdef HEAT_EQ
		farray_r[TH][i] = 1.0 +  0.5 * (tanh((x[i] + 0.375) / a) - tanh((x[i] + 0.125) / a)) + 0.5 * (tanh((x[i] - 0.125) / a) - tanh((x[i] - 0.375) / a));
	#endif
	#ifdef INCOMPRESSIBLE
		// farray_r[VX][i] = 1.0 ;
		// // // farray_r[VY][i] = 1.0 * sin(2.0*M_PI*x[i]);
		// farray_r[VY][i] = 2.0;
		// farray_r[VZ][i] = 0.0;
		// Taylor - Green vortex
		farray_r[VX][i] =   sin(2.0*M_PI*x[i]/param.lx) * cos(2.0*M_PI*y[i]/param.ly);
		// farray_r[VY][i] = 1.0 * sin(2.0*M_PI*x[i]);
		farray_r[VY][i] = - cos(2.0*M_PI*x[i]/param.lx) * sin(2.0*M_PI*y[i]/param.ly);
		farray_r[VZ][i] = 0.0;
	#endif
	#ifdef BOUSSINESQ
		farray_r[TH][i] = 0.0;
	#endif
	}

// 	for (int i = 0; i < 10; i++){
//
// 		std::printf("x[%d] = %.2e \t y[%d] = %.2e \t z[%d] = %.2e \t th[%d] = %.2e \t",i,x[i],i,y[i],i,z[i],i,farray_r[TH][i]);
//
// 	}

	// int idx;
	// #ifdef HEAT_EQ
	// for(i = 0 ; i < nx ; i++) {
	// 	idx = (nz + 2) * ny * i;
	// 	std::printf("x[%d] = %.2e \t  th[%d] = %.2e \n",idx,x[idx],idx,farray_r[TH][idx]);
	// }
	// #endif
 //
 //
 //

	free(x);
	free(y);
	free(z);

	std::printf("Finished initializing spatial structure\n");
}
