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

	double R2 = 0.0;
	double sigma  = 200.;
	double v0 = 0.05;
	double u0 = 0.05;


	for (int i = 0; i < 2*ntotal_complex; i++){

		// Dinshaw S. Balsara 2004 ApJS 151 149 DOI 10.1086/381377

		R2 = x[i]*x[i] + y[i]*y[i];

		farray_r[VX][i] = v0 - 1./(2.0 * M_PI) * exp( (1.0  - sigma*R2)/2.0 ) * y[i] ;
		farray_r[VY][i] = u0 + 1./(2.0 * M_PI) * exp( (1.0  - sigma*R2)/2.0 ) * x[i] ;
		farray_r[VZ][i] = 0.0 ;

		farray_r[BX][i] = - 1./(2.0 * M_PI) * exp( (1.0  - sigma*R2)/2.0 ) * y[i] ;
		farray_r[BY][i] =   1./(2.0 * M_PI) * exp( (1.0  - sigma*R2)/2.0 ) * x[i] ;
		farray_r[BZ][i] = 0.0 ;


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
