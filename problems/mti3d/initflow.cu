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

	double sigma  = 0.9341413811120219;
	double Pe     = param.reynolds_ani;
    double Reeta  = param.reynolds_m;
    double kparallel  = (2.0*M_PI/param.lx)*12.0;
	double B0 = 1e-4;

	for (int i = 0; i < 2*ntotal_complex; i++){

		// MTI eigenmode

		farray_r[VX][i] = 0.0 ;
		farray_r[VY][i] = 0.0 ;
		farray_r[VZ][i] = -0.00001*sin(kparallel*x[i]);

		farray_r[BX][i] = B0 ;
		farray_r[BY][i] = 0.0 ;
		farray_r[BZ][i] = -0.00001*cos(kparallel*x[i])*B0*kparallel/(sigma+kparallel*kparallel/Reeta);

		farray_r[TH][i] = 1.0/(sigma + kparallel*kparallel/Pe)*(param.N2 - kparallel*kparallel/Pe/(sigma+kparallel*kparallel/Reeta) ) * farray_r[VZ][i];


	}


	free(x);
	free(y);
	free(z);

	std::printf("Finished initializing spatial structure\n");
}
