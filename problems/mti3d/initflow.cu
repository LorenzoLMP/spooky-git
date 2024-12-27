#include "common.hpp"
#include "cufft_routines.hpp"
// #include "cuda_kernels.hpp"
#include "fields.hpp"
#include "parameters.hpp"

void Fields::initSpatialStructure(){

	std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;

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

	double sigma  = 0.9341413811120219;
	double Pe     = param_ptr->reynolds_ani;
 	double Reeta  = param_ptr->reynolds_m;
    	double kparallel  = (2.0*M_PI/param_ptr->lx)*12.0;
	double B0 = 1e-4;

	for (int i = 0; i < 2*grid.NTOTAL_COMPLEX; i++){

		// MTI eigenmode

		farray_r[vars.VX][i] =  0.01*cos(0.5*kparallel*y[i])+0.02*sin(kparallel*z[i]/6.0);
		farray_r[vars.VY][i] =  0.0001*sin(0.25*kparallel*z[i])-0.003*cos(kparallel*x[i]/6.0);
		farray_r[vars.VZ][i] = -0.001*sin(kparallel*x[i])+0.004*cos(0.5*kparallel*y[i]);

		farray_r[vars.BX][i] = B0 ;
		farray_r[vars.BY][i] = 0.0001*sin(kparallel*z[i]/4.0) ;
		// farray_r[vars.BZ][i] = -0.001*cos(kparallel*x[i])*B0*kparallel/(sigma+kparallel*kparallel/Reeta);
		farray_r[vars.BZ][i] = 0.00;

		// farray_r[vars.TH][i] = 1.0/(sigma + kparallel*kparallel/Pe)*(param_ptr->N2 - kparallel*kparallel/Pe/(sigma+kparallel*kparallel/Reeta) ) * farray_r[vars.VZ][i];
		farray_r[vars.TH][i] = 0.004*sin(kparallel*x[i]/4.)-0.00003*cos(kparallel*y[i]/2.0);


	}


	free(x);
	free(y);
	free(z);

	std::printf("Finished initializing spatial structure\n");
}
