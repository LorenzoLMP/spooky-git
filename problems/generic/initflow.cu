#include "define_types.hpp"
#include "cufft_routines.hpp"
// #include "cuda_kernels.hpp"
#include "fields.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "parameters.hpp"

void Fields::init_SpatialStructure(Parameters &param){

	int i,j,k;

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
				x[k + (grid.NZ + 2) * j + (grid.NZ + 2) * grid.NY * i] = - param.lx / 2 + (param.lx * i) / grid.NX;
				y[k + (grid.NZ + 2) * j + (grid.NZ + 2) * grid.NY * i] = - param.ly / 2 + (param.ly * j ) / grid.NY;
				z[k + (grid.NZ + 2) * j + (grid.NZ + 2) * grid.NY * i] = - param.lz / 2 + (param.lz * k ) / grid.NZ;
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
				x[k + (grid.NZ) * j + (grid.NZ) * (grid.NY + 2) * i] = - param.lx / 2 + (param.lx * i) / grid.NX;
				y[k + (grid.NZ) * j + (grid.NZ) * (grid.NY + 2) * i] = - param.ly / 2 + (param.ly * j ) / grid.NY;
				z[k + (grid.NZ) * j + (grid.NZ) * (grid.NY + 2) * i] = - param.lz / 2 + (param.lz * k ) / grid.NZ;
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
	double a = 0.01;
	for (int i = 0; i < 2*grid.NTOTAL_COMPLEX; i++){
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
	// for(i = 0 ; i < grid.NX ; i++) {
	// 	idx = (grid.NZ + 2) * grid.NY * i;
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
