#include "define_types.hpp"
#include "fields.hpp"
#include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
#include "cublas_routines.hpp"
#include "cuda_kernels.hpp"
#include "parameters.hpp"
#include "timestepping.hpp"
// #include "wavevector.hpp"
#include "supervisor.hpp"


Fields::~Fields() {
    free(all_fields);
    // the following can be commented out for production
    free(all_dfields);

    free(farray);
    free(farray_r);
    // the following 2 can be commented out for production
    free(dfarray);
    free(dfarray_r);
    free(d_farray);
    free(d_farray_r);

    free(d_dfarray);
    free(d_dfarray_r);

    free(d_farray_buffer_r);

    free(d_tmparray);

}

// void Fields::init_Fields( int num, Parameters *p_in )  {
Fields::Fields(Supervisor &sup_in, Parameters &p_in) : wavevector(p_in.lx, p_in.ly, p_in.lz) {

    supervisor_ptr = &sup_in;
    num_fields = NUM_FIELDS;
    std::printf("num_fields: %d \n",num_fields);

    num_tmp_array = 6; // need to check again why we need all these tmp arrays
    std::printf("num_tmp_array: %d \n",num_tmp_array);

    // current_dt = 0.0;
    // current_time = 0.0;
    // current_step = 0;
    // t_lastsnap = 0.0;
    // t_lastvar = 0.0;
    // num_save = 0;

    // param = &p_in;
    // timestep = timestep_in;
    // r_data = (scalar_type *) malloc( sizeof( scalar_type ) * 2*ntotal_complex ) ;
    // c_data = (data_type *) r_data;

    // previous implementation
    all_fields = (data_type *) malloc( (size_t) sizeof(data_type) * ntotal_complex * num_fields);
    // the following can be commented out for production
    all_dfields = (data_type *) malloc( (size_t) sizeof(data_type) * ntotal_complex * num_fields);

    // holds array of complex-valued arrays (the fields)
    farray = (data_type **) malloc( (size_t) sizeof(data_type *) * num_fields);
    // the following can be commented out for production
    dfarray = (data_type **) malloc( (size_t) sizeof(data_type *) * num_fields);

    // holds array of real-valued arrays (the fields)
    farray_r = (scalar_type **) malloc( (size_t) sizeof(data_type *) * num_fields);
    // the following can be commented out for production
    dfarray_r = (scalar_type **) malloc( (size_t) sizeof(data_type *) * num_fields);

    // these are arrays of arrays on host that contain address in device memory
    d_farray    = (data_type **) malloc( (size_t) sizeof(data_type *) * num_fields);
    d_farray_r = (scalar_type **) malloc( (size_t) sizeof(data_type *) * num_fields);
    d_dfarray   = (data_type **) malloc( (size_t) sizeof(data_type *) * num_fields);
    d_dfarray_r = (scalar_type **) malloc( (size_t) sizeof(data_type *) * num_fields);
    d_tmparray  = (data_type **) malloc( (size_t) sizeof(data_type *) * num_tmp_array);
    d_tmparray_r = (scalar_type **) malloc( (size_t) sizeof(data_type *) * num_tmp_array);
    // this is the mega array that contains temporary scratch arrays
    // d_tmparray = (data_type **) malloc( (size_t) sizeof(data_type *) * num_tmp_array);

    // this is a buffer array that is reserved for the real fields
    d_farray_buffer_r = (scalar_type **) malloc( (size_t) sizeof(data_type *) * num_fields);


    // initialize field arrays and dfield arrays (which are arrays of arrays), farray[0] = vx, farray[1] = vy, etc...
    // this is only for cpu arrays
    for (int i = 0 ; i < num_fields ; i++) {
        farray[i]   = all_fields + i*ntotal_complex;
        farray_r[i] = (scalar_type *) farray[i];
        // the following 2 can be commented out for production
        // dfarray[i]  = all_dfields + i*ntotal_complex;
        // dfarray_r[i] = (scalar_type *) dfarray[i];
    }
    // std::printf("before init wavevec lx = %f \t ly = %f \t lz = %f\n",param->lx, param->ly, param->lz);
    // Wavevector wavevector(param->lx, param->ly, param->lz);

    // wavevector.init_Wavevector(param->lx, param->ly, param->lz);
    // vx_r = (scalar_type *) vx;
    // vy_r = (scalar_type *) vy;
    std::printf("Initializing spatial structure\n");
    for (int i = 0; i < ntotal_complex * num_fields; i++){
        all_fields[i] = data_type(0.0,0.0);
        all_dfields[i] = data_type(0.0,0.0);
    }

    init_SpatialStructure(p_in);
    // wavevector.init_Wavevector(p_in);
    // std::printf("num_fields: %d \n",num_fields);
    // for (int n = 0 ; n < num_fields ; n++) {
    //     std::printf("n=%d:\n",n);
    //     for (int i = 0; i < 2*ntotal_complex; i++){
    //         farray_r[n][i] = i;
    //         dfarray_r[n][i] = i;
    //     }
    // }


    ////////////////////////////////////////////////////////
    // useful for benchmarks
    ////////////////////////////////////////////////////////
    /*
    unsigned int idx;
    for (int n = 0 ; n < num_fields ; n++) {
        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){
                for (int k = 0; k < nz; k++){
                    idx = k + (nz/2+1)*2 * ( j + i * ny);
                    // std::printf("idx=%d:\n",idx);
                    // useful for benchmarks
                    // farray_r[n][idx]  = ((double) n+1.) * (fmod((double)idx, 2.) - 1.);
                    // dfarray_r[n][idx]  = ((double) n+1.) * (fmod((double)idx, 2.) - 1.);
                    // farray_r[n][idx]  = 1.0;
                    // dfarray_r[n][idx]  = 1.0;
                    farray_r[n][idx]  = k + nz * ( j + i * ny);
                    dfarray_r[n][idx]  = k + nz * ( j + i * ny);
                }
            }
        }
    }
    for (int n = 0 ; n < num_fields ; n++) {
        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){
                for (int k = nz; k < nz+2; k++){
                    idx = k + (nz/2+1)*2 * ( j + i * ny);
                    farray_r[n][idx]  = 0.0;
                    dfarray_r[n][idx] = 0.0;
                }
            }
        }
    }
    */






    // for (int i = 0; i < nx; i++){
    //     for (int j = 0; j < ny; j++){
    //         for (int k = 0; k < nz; k++){
    //             idx = k + (nz/2+1)*2 * ( j + i * ny);
    //             // std::printf("idx=%d:\n",idx);
    //             farray_r[0][idx]  = 1.0;
    //             farray_r[1][idx]  = 1.0+idx;
    //             farray_r[2][idx]  = 1./(1.0+idx);
    //             // dfarray_r[n][idx] = 0.0;
    //         }
    //     }
    // }
    // for (int n = 0 ; n < num_fields ; n++) {
    //     for (int i = 0; i < nx; i++){
    //         for (int j = 0; j < ny; j++){
    //             for (int k = nz; k < nz+2; k++){
    //                 idx = k + (nz/2+1)*2 * ( j + i * ny);
    //                 farray_r[n][idx]  = 0.0;
    //                 // dfarray_r[n][idx] = 0.0;
    //             }
    //         }
    //     }
    // }

    // for (int i = 0; i < ntotal_complex; i++){
    //     farray[0][i] = data_type(1.0,0.0);
    //     farray[1][i] = data_type(0.0,1.0);
    // }

}

void Fields::print_host_values() {

    unsigned int idx;
    std::printf("Printing host array:\n");
    // for (int i = 0; i < 2; i++){
    //     for (int j = 0; j < 2; j++){
    //         for (int k = 0; k < 2; k++){
    //             idx = k + (nz/2+1)*2 * ( j + i * ny);
    //             // std::printf("v1[%d]= %f \t v2[%d]= %f \n", idx, farray_r[0][idx], idx, farray_r[1][idx]);
    //             for (int n = 0; n < num_fields; n++){
    //                 std::printf("v[%d][%d] = %.3e \t", n, idx, farray_r[n][idx]);
    //             }
    //             std::cout << std::endl;
    //         }
    //     }
    // }
    for (int i = 0; i < 25; i++){
        idx =  (nz/2+1)*2 * ( i * ny);
        // std::printf("v1[%d]= %f \t v2[%d]= %f \n", idx, farray_r[0][idx], idx, farray_r[1][idx]);
        for (int n = 0; n < num_fields; n++){
            std::printf("v[%d][%d] = %.7e \t", n, idx, farray_r[n][idx]);
        }
        std::cout << std::endl;
    }

    // for (int i = 0; i < 8; i++){
    //     std::printf("vx[%d] = %f \t vy[%d] = %f \n", i, farray_bis_r[0][i],i, farray_bis_r[1][i]);
    //     // std::printf("vy[%d] %f \n", i, farray_r[1][i]);
    // }
}

void Fields::print_device_values() {
    data_type *all_fields_bis;
    // data_type *all_dfields_bis;
    all_fields_bis = (data_type *) malloc( (size_t) sizeof(data_type) * ntotal_complex * num_fields);
    // all_dfields_bis = (data_type *) malloc( (size_t) sizeof(data_type) * ntotal_complex * num_fields);


    for (int i = 0; i < ntotal_complex * num_fields; i++){
        all_fields_bis[i] = data_type(0.0,0.0);
        // all_dfields_bis[i] = data_type(0.0,0.0);
    }

    // // transform from complex to real before moving over to host

    // for (int n = 0 ; n < num_fields ; n++) {
    //     c2r_fft(d_farray[n], d_farray_r[n]);
    //     c2r_fft(d_dfarray[n], d_dfarray_r[n]);
    // }

    CUDA_RT_CALL(cudaMemcpy(all_fields_bis, d_all_fields, sizeof(data_type) * ntotal_complex * num_fields, cudaMemcpyDeviceToHost));
    // CUDA_RT_CALL(cudaMemcpy(all_dfields_bis, d_all_dfields, sizeof(data_type) * ntotal_complex * num_fields, cudaMemcpyDeviceToHost));

    // // transform from real to complex to resume code execution
    // for (int n = 0 ; n < num_fields ; n++) {
    //     r2c_fft(d_farray_r[n], d_farray[n]);
    //     r2c_fft(d_dfarray_r[n], d_dfarray[n]);
    // }

    data_type **farray_bis;
    // data_type **dfarray_bis;
    // scalar_type **farray_bis_r, **dfarray_bis_r;
    farray_bis = (data_type **) malloc( (size_t) sizeof(data_type *) * num_fields);
    // farray_bis_r = (scalar_type **) malloc( (size_t) sizeof(data_type *) * num_fields);
    // dfarray_bis = (data_type **) malloc( (size_t) sizeof(data_type *) * num_fields);
    // dfarray_bis_r = (scalar_type **) malloc( (size_t) sizeof(data_type *) * num_fields);

    for (int i = 0 ; i < num_fields ; i++) {
        farray_bis[i]   = all_fields_bis + i*ntotal_complex;
        // farray_bis_r[i] = (scalar_type *) farray_bis[i];
        // dfarray_bis[i]   = all_dfields_bis + i*ntotal_complex;
        // dfarray_bis_r[i] = (scalar_type *) dfarray_bis[i];
    }

    unsigned int idx;
    std::printf("Values from device farray:\n");
    // for (int i = 0; i < 25; i++){
    //     idx =  (nz/2+1)*2 * ( i * ny);
    //     // std::printf("v1[%d]= %f \t v2[%d]= %f \n", idx, farray_r[0][idx], idx, farray_r[1][idx]);
    //     for (int n = 0; n < num_fields; n++){
    //         std::printf("v[%d][%d] = %.7e \t", n, idx, farray_bis_r[n][idx]);
    //         std::printf("dv[%d][%d] = %.7e \t", n, idx, dfarray_bis_r[n][idx]);
    //     }
    //     std::cout << std::endl;
    // }

    for (int n = 0; n < num_fields; n++){
        for (int i = 0; i < 2; i++){
            for (int j = 0; j < 2; j++){
                for (int k = 0; k < 2; k++){
                    idx = k + (nz/2+1) * ( j + i * ny);
                    // std::printf("v1[%d]= %f \t v2[%d]= %f \n", idx, farray_bis_r[0][idx], idx, farray_bis_r[1][idx]);
                    std::printf("(i,j,k) = (%02d,%02d,%02d), idx = %06d\t",i,j,k, idx);

                    std::printf("v[%01d] = %.2e + 1j %.2e  ", n,  farray_bis[n][idx].real(),farray_bis[n][idx].imag());
                    std::cout << std::endl;
                    // std::printf("dv[%01d] = %.2e \t", n,  dfarray_bis_r[n][idx]);
                }

            }
        }
    }

    free(all_fields_bis);
    free(farray_bis);

}
void Fields::allocate_and_move_to_gpu() {
    std::printf("Allocating to gpu:\n");
    // void *d_vx, *d_vy;
    // this is the mega array that contains fields
    CUDA_RT_CALL(cudaMalloc(&d_all_fields, (size_t) sizeof(data_type) * ntotal_complex * num_fields));
    // this is the mega array that contains dfields
    CUDA_RT_CALL(cudaMalloc(&d_all_dfields, (size_t) sizeof(data_type) * ntotal_complex * num_fields));
    // this is the mega array that contains temporary scratch arrays
    CUDA_RT_CALL(cudaMalloc(&d_all_tmparray, (size_t) sizeof(data_type) * ntotal_complex * num_tmp_array));
    // this is the buffer array for real fields
    CUDA_RT_CALL(cudaMalloc(&d_all_buffer_r, (size_t) sizeof(data_type) * ntotal_complex * num_fields));

    // copy to gpu

    CUDA_RT_CALL(cudaMemcpy(d_all_fields, all_fields, (size_t) sizeof(data_type) * ntotal_complex * num_fields, cudaMemcpyHostToDevice));
    // this shouldn't be necessary
    CUDA_RT_CALL(cudaMemcpy(d_all_dfields, all_dfields, (size_t) sizeof(data_type) * ntotal_complex * num_fields, cudaMemcpyHostToDevice));

    CUDA_RT_CALL(cudaMemcpy(d_all_tmparray, all_dfields, (size_t) sizeof(data_type) * ntotal_complex * num_fields, cudaMemcpyHostToDevice));


    std::printf("num_fields: %d \n",num_fields);

    for (int i = 0 ; i < num_fields ; i++) {
        d_farray[i]  = d_all_fields + i*ntotal_complex;
        d_dfarray[i] = d_all_dfields + i*ntotal_complex;
        // d_dfarraytmp[i] = d_all_dfieldstmp + i*ntotal_complex;
        d_farray_r[i] = (scalar_type *) d_farray[i];
        d_dfarray_r[i] = (scalar_type *) d_dfarray[i];

        // this is the farray buffer array for real fields
        d_farray_buffer_r[i] = (scalar_type *) ( d_all_buffer_r + i*ntotal_complex );
    }

    for (int i = 0 ; i < num_tmp_array ; i++) {
        d_tmparray[i]  = d_all_tmparray + i*ntotal_complex;
        d_tmparray_r[i] = (scalar_type *) d_tmparray[i];
    }
    std::printf("finished filling of device array of arrays\n");

    // transform from real to complex to begin code execution
    for (int n = 0 ; n < num_fields ; n++) {
        r2c_fft(d_farray_r[n], d_farray[n]);
        r2c_fft(d_dfarray_r[n], d_dfarray[n]);
    }

    // printf("Address stored in ip variable:", d_all_fields );

    // for (int i = 0 ; i < num_fields ; i++) {
    //     d_farray[i]   = d_all_fields + i*ntotal_complex;
    //     // d_farray_r[i] = (scalar_type *) d_farray[i];
    //     // d_dfarray[i]  = d_all_dfields + i*ntotal_complex;
    //     // d_dfarray_r[i] = (scalar_type *) d_dfarray[i];
    // }

    // for (int i = 0; i < 10; i++){
    //     std::printf("vx_r[%d] %f \n", i, farray_r[0][i]);
    //     // std::printf("d_vx_r[%d] %f \n", i, d_farray_r[0][i]);
    // }

    wavevector.allocate_and_move_to_gpu();
    std::printf("Finished allocating wavevector and move to gpu\n");
}


void Fields::copy_back_to_host() {

    for (int n = 0 ; n < num_fields ; n++) {
        c2r_fft(d_farray[n], d_farray_r[n]);
    }

    CUDA_RT_CALL(cudaMemcpy(all_fields, d_all_fields, sizeof(data_type) * ntotal_complex * num_fields, cudaMemcpyDeviceToHost));

    for (int n = 0 ; n < num_fields ; n++) {
        r2c_fft(d_farray_r[n], d_farray[n]);
    }


    std::printf("array copied back from device \n");


    wavevector.sync_with_host();
}

void Fields::compare_with_original(){
    data_type *all_fields_bis;
    all_fields_bis = (data_type *) malloc( (size_t) sizeof(data_type) * ntotal_complex * num_fields);
    for (int i = 0; i < ntotal_complex * num_fields; i++){
        all_fields_bis[i] = data_type(0.0,0.0);
    }

    // transform from complex to real before moving to host
    for (int n = 0 ; n < num_fields ; n++) {
        c2r_fft(d_farray[n], d_farray_r[n]);
    }

    CUDA_RT_CALL(cudaMemcpy(all_fields_bis, d_all_fields, sizeof(data_type) * ntotal_complex * num_fields, cudaMemcpyDeviceToHost));

    // transform from real to complex to resume code execution
    for (int n = 0 ; n < num_fields ; n++) {
        r2c_fft(d_farray_r[n], d_farray[n]);
    }
    data_type **farray_bis;
    scalar_type **farray_bis_r;
    farray_bis = (data_type **) malloc( (size_t) sizeof(data_type *) * num_fields);
    farray_bis_r = (scalar_type **) malloc( (size_t) sizeof(data_type *) * num_fields);


    for (int i = 0 ; i < num_fields ; i++) {
        farray_bis[i]   = all_fields_bis + i*ntotal_complex;
        farray_bis_r[i] = (scalar_type *) farray_bis[i];
    }

    std::printf("array copied back from device \n");
    // verify results
    double error{};
    double ref{};
    double diff = 0.0;
    // int idx;
    unsigned int idx;

    // for (int i = 0; i < ntotal_complex * num_fields; i++){
    //     std::printf("index = %d, v_old = %lf + I * %lf, v_new = %lf + I * %lf \n",i,all_fields[i].real(),all_fields[i].imag(),all_fields_bis[i].real(),all_fields_bis[i].imag());
    //     error += std::norm(all_fields[i] - all_fields_bis[i]);
    //     diff = std::norm(all_fields[i] - all_fields_bis[i]);
    //     if (diff > 0.1 * std::norm(all_fields[i])) {
    //         // std::printf("index = %d, v_old = %lf + I * %lf, v_new = %lf + I * %lf \n",i,all_fields[i].real(),all_fields[i].imag(),all_fields_bis[i].real(),all_fields_bis[i].imag());
    //         break;
    //     }
    //     ref += std::norm(all_fields_bis[i]);
    // }
    for ( int n = 0 ; n < num_fields ; n++) {
        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){
                for (int k = 0; k < nz; k++){
                    idx = k + (nz/2+1)*2 * ( j + i * ny);

                    diff = std::norm(farray_r[n][idx] - farray_bis_r[n][idx]);
                    // if (diff > 0.1 * std::norm(farray_r[n][idx])) {
                    //     std::printf("index = %d, v_old = %lf , v_new = %lf \n",i,farray_r[n][idx],farray_bis_r[n][idx]);
                    // }
                    error += std::norm(farray_r[n][idx] - farray_bis_r[n][idx]);
                    ref += std::norm(farray_bis_r[n][idx]);
                }
            }
        }
    }


    double l2_error = (ref == 0.0) ? std::sqrt(error) : std::sqrt(error) / std::sqrt(ref);
    if (l2_error < 0.001) {
        std::cout << "PASSED with L2 error = " << l2_error << std::endl;
    } else {
        std::cout << "FAILED with L2 error = " << l2_error << std::endl;
    };
    // CUDA_RT_CALL(cudaMemcpy(vy, d_vy, sizeof(data_type) * ntotal_complex, cudaMemcpyDeviceToHost));

    // for (int i = 0; i < 8; i++){
    //     std::printf("index = %d, v_old = %lf , v_new = %lf \n",i,farray_r[0][i],farray_bis_r[0][i]);// std::printf("idx=%d:\n",idx);
    // }

    std::printf("Sample values:\n");
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++){
            for (int k = 0; k < 2; k++){
                idx = k + (nz/2+1)*2 * ( j + i * ny);
                std::printf("vold[%d]= %f \t vnew[%d]= %f \n", idx, farray_r[0][idx], idx, farray_bis_r[0][idx]);
            }
        }
    }

    free(all_fields_bis);
    free(farray_bis);
    free(farray_bis_r);

}
void Fields::clean_gpu(){
    CUDA_RT_CALL(cudaFree(d_all_fields));
    CUDA_RT_CALL(cudaFree(d_all_dfields));
    CUDA_RT_CALL(cudaFree(d_all_tmparray));

    CUDA_RT_CALL(cudaFree(d_all_buffer_r));

    // CUDA_RT_CALL(cudaFree(d_farray));
    // CUDA_RT_CALL(cudaFree(d_dfarray));
    // CUDA_RT_CALL(cudaFree(d_vy));

    wavevector.clean_gpu();
}

// void Fields::Complex2RealFields(data_type* ComplexField_in, scalar_type* RealField_out, int num_fields){
//
//
//     // assign fields to [num_fields] tmparray (memory block starts at d_all_tmparray)
//     int blocksPerGrid = ( num_fields * ntotal_complex + threadsPerBlock - 1) / threadsPerBlock;
//     ComplexVecAssign<<<blocksPerGrid, threadsPerBlock>>>(ComplexField_in, (data_type*) RealField_out, num_fields * ntotal_complex);
//
//     // compute FFTs from complex to real fields
//     for (int n = 0; n < num_fields; n++){
//         c2r_fft((data_type*) RealField_out + n * ntotal_complex,  RealField_out + n * 2*ntotal_complex, supervisor_ptr);
//     }
//
// }
