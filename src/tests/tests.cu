#include <array>
#include <complex>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include "cufft_routines.hpp"
#include <cuda_runtime.h>
#include "common.hpp"
#include "cublas_routines.hpp"
// #include "cuda_kernels.hpp"
#include "cuda_kernels_generic.hpp"
// #include <cuda.h>




int test_forward_inverse_transform(){
    // size_t grid.NX = 256;
    // size_t grid.NY = 128;
    // size_t grid.NZ = 64;
    // dim_t  grid.FFT_SIZE = {grid.NX, grid.NY, grid.NZ};
    // size_t ntotal = grid.FFT_SIZE[0] * grid.FFT_SIZE[1] * grid.FFT_SIZE[2];
    // size_t grid.NTOTAL_COMPLEX = grid.FFT_SIZE[0] * grid.FFT_SIZE[1] * ((grid.FFT_SIZE[2] / 2) + 1);
    int Niter = 50;
    // extern int dimGrid, dimBlock;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int success = 1; // fail



    cpudata_t cpu_r_data((size_t) 2*grid.NTOTAL_COMPLEX);
    cpudata_t cpu_r_data_out((size_t) 2*grid.NTOTAL_COMPLEX);

    unsigned int idx;
    for (int i = 0; i < grid.NX; i++){
        for (int j = 0; j < grid.NY; j++){
            for (int k = 0; k < grid.NZ; k++){
                idx = k + (grid.NZ/2+1)*2 * ( j + i * grid.NY);
                cpu_r_data[idx] = idx;
            }
        }
    }
    for (int i = 0; i < grid.NX; i++){
        for (int j = 0; j < grid.NY; j++){
            for (int k = grid.NZ; k < grid.NZ+2; k++){
                idx = k + (grid.NZ/2+1)*2 * ( j + i * grid.NY);
                cpu_r_data[idx] = 0.0;
            }
        }
    }

    std::printf("Input array:\n");
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++){
            for (int k = 0; k < 2; k++){
                idx = k + (grid.NZ/2+1)*2 * ( j + i * grid.NY);
                std::printf("v[%d] %f \n", idx, cpu_r_data[idx]);
            }
        }
    }
    std::printf("=====\n");


    // Create device data arrays
    void *c_data;
    CUDA_RT_CALL(cudaMalloc(&c_data, (size_t) sizeof(data_type) * grid.NTOTAL_COMPLEX));
    std::printf("array size (in MiB): %f \n",(float) (sizeof(data_type) * grid.NTOTAL_COMPLEX/1e6));
    // Create pointer to complex array to store real data
    cufftDoubleReal *r_data = (scalar_type *) c_data;
    // Copy input data to GPUs
    CUDA_RT_CALL(cudaMemcpy(r_data, cpu_r_data.data(), (size_t) sizeof(scalar_type) * 2 * grid.NTOTAL_COMPLEX, cudaMemcpyHostToDevice));

    // init plans
    init_plan(grid.FFT_SIZE);

    cudaEventRecord(start);
    // Do forward and inverse transform
    for (int ii = 0; ii < Niter; ii++) {
        r2c_fft(r_data, c_data);
        c2r_fft(c_data, r_data);
    }
    cudaEventRecord(stop);

    // Copy output data to CPU
    CUDA_RT_CALL(cudaMemcpy(cpu_r_data_out.data(), r_data, sizeof(scalar_type) * 2 * grid.NTOTAL_COMPLEX, cudaMemcpyDeviceToHost));


    std::printf("Output array:\n");
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++){
            for (int k = 0; k < 2; k++){
                idx = k + (grid.NZ/2+1)*2 * ( j + i * grid.NY);
                std::printf("v[%d] %f \n", idx, cpu_r_data_out[idx]);
            }
        }
    }
    std::printf("=====\n");

    CUDA_RT_CALL(cudaFree(c_data));

    finish_cufft();


    // verify results
    double error{};
    double ref{};
    for (int i = 0; i < grid.NX; i++){
        for (int j = 0; j < grid.NY; j++){
            for (int k = 0; k < grid.NZ; k++){
                idx = k + (grid.NZ/2+1)*2 * ( j + i * grid.NY);
                error += std::norm(cpu_r_data[idx] - cpu_r_data_out[idx]);
                ref += std::norm(cpu_r_data_out[idx]);
            }
        }
    }

    double l2_error = (ref == 0.0) ? std::sqrt(error) : std::sqrt(error) / std::sqrt(ref);
    if (l2_error < 1e-12) {
        std::cout << "PASSED with L2 error = " << l2_error << std::endl;
        success = 0; // success
    } else {
        std::cout << "FAILED with L2 error = " << l2_error << std::endl;
    };

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::printf("Elapsed time (in s): %.5f \t Approx time per FFT (in ms): %.5f \n",milliseconds/1000, 0.5*milliseconds/Niter);
    float gflops = 1e-9*5*(grid.NTOTAL_COMPLEX)*log2(grid.NTOTAL_COMPLEX)/(0.5*1e-3*milliseconds/Niter);
    std::printf("Average GFlop/s (per Fourier transform) %.2f\n",gflops);
    

    return success;
};

/*
void test_do_multiplications() {
    // init_plan(grid.FFT_SIZE);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // cudaError_t devsyncherr;

    // Do forward and inverse transform
    int Niter=100;

    cpudata_t cpu_r_data1((size_t) 2*grid.NTOTAL_COMPLEX);
    cpudata_t cpu_r_data2((size_t) 2*grid.NTOTAL_COMPLEX);

    unsigned int idx;
    for (int i = 0; i < grid.NX; i++){
        for (int j = 0; j < grid.NY; j++){
            for (int k = 0; k < grid.NZ; k++){
                idx = k + (grid.NZ/2+1)*2 * ( j + i * grid.NY);
                cpu_r_data1[idx] = idx;
                cpu_r_data2[idx] = 2.0*idx;
            }
        }
    }
    for (int i = 0; i < grid.NX; i++){
        for (int j = 0; j < grid.NY; j++){
            for (int k = grid.NZ; k < grid.NZ+2; k++){
                idx = k + (grid.NZ/2+1)*2 * ( j + i * grid.NY);
                cpu_r_data1[idx] = 0.0;
                cpu_r_data2[idx] = 0.0;
            }
        }
    }
    cpu_r_data1[0] = 1.0;
    cpu_r_data2[0] = 1.0;

    std::printf("Input array:\n");
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++){
            for (int k = 0; k < 2; k++){
                idx = k + (grid.NZ/2+1)*2 * ( j + i * grid.NY);
                std::printf("v1[%d]= %f \t v2[%d]= %f \n", idx, cpu_r_data1[idx], idx, cpu_r_data2[idx]);
                // std::printf("v2[%d] %f \n", idx, cpu_r_data2[idx]);
            }
        }
    }
    std::printf("=====\n");

    // Create device data arrays
    scalar_type *dev_data1;
    scalar_type *dev_data2;
    CUDA_RT_CALL(cudaMalloc(&dev_data1, (size_t) sizeof(scalar_type) * 2 * grid.NTOTAL_COMPLEX));
    CUDA_RT_CALL(cudaMalloc(&dev_data2, (size_t) sizeof(scalar_type) * 2 * grid.NTOTAL_COMPLEX));
    std::printf("array size (in MiB): %f \n",(float) (sizeof(scalar_type) * 2 * grid.NTOTAL_COMPLEX/1e6));

    // Copy input data to GPUs
    CUDA_RT_CALL(cudaMemcpy(dev_data1, cpu_r_data1.data(), (size_t) sizeof(scalar_type) * 2 * grid.NTOTAL_COMPLEX, cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(dev_data2, cpu_r_data2.data(), (size_t) sizeof(scalar_type) * 2 * grid.NTOTAL_COMPLEX, cudaMemcpyHostToDevice));

    float milliseconds = 0;
    int numElements = 2*grid.NTOTAL_COMPLEX;
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

/////////////////////////////////////////////////////
    // using THRUST
/////////////////////////////////////////////////////

    cudaEventRecord(start);
    for (int ii = 0; ii < Niter; ii++) {
        // std::printf("iter %d \n", ii);
        // thrust::transform(thrust::device_pointer_cast(d_farray_r[0]), thrust::device_pointer_cast(d_farray_r[0])+2*grid.NTOTAL_COMPLEX, thrust::device_pointer_cast(d_farray_r[1]), thrust::device_pointer_cast(d_farray_r[0]), thrust::multiplies<scalar_type>());
        // for (int n = 0 ; n < num_fields ; n++) {
        //     r2c_fft(d_farray_r[n], d_farray[n]);
        //     // c2r_fft(d_farray[n], d_farray_r[n]);
        // }

        // this operation does pointwise v1*v2 operation and stores the result in v1
        thrust::transform(thrust::device_pointer_cast(dev_data1), thrust::device_pointer_cast(dev_data1)+2*grid.NTOTAL_COMPLEX, thrust::device_pointer_cast(dev_data2), thrust::device_pointer_cast(dev_data1), thrust::multiplies<scalar_type>());
        // divide back
        thrust::transform(thrust::device_pointer_cast(dev_data1), thrust::device_pointer_cast(dev_data1)+2*grid.NTOTAL_COMPLEX, thrust::device_pointer_cast(dev_data2), thrust::device_pointer_cast(dev_data1), thrust::divides<scalar_type>());


    }
    cudaEventRecord(stop);

    cudaDeviceSynchronize();
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::printf("Thrust elapsed time (in s): %.5f \t Approx time per multiply (in ms): %.5f \n",milliseconds/1000, milliseconds/Niter);


    ///////////////////////////////////////////////////
    // same operation as before but with CUBLAS
    ///////////////////////////////////////////////////

    init_cublas();
    cublasStatus_t stat;
    scalar_type *scratch;
    CUDA_RT_CALL(cudaMalloc(&scratch, (size_t) sizeof(scalar_type) * 2 * grid.NTOTAL_COMPLEX));
    // cublasSideMode_t mode = CUBLAS_SIDE_RIGHT;
    // int n = 2*grid.NTOTAL_COMPLEX;
    // int m = 1;
    // int lda = 2*grid.NTOTAL_COMPLEX;
    // int incx = 1;
    // int ldc = 2*grid.NTOTAL_COMPLEX;

    cublasSideMode_t mode = CUBLAS_SIDE_LEFT;
    int m = numElements;
    int n = 1;
    int lda = numElements;
    int incx = 1;
    int ldc = numElements;

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
            threadsPerBlock);
    cudaEventRecord(start);
    for (int ii = 0; ii < Niter; ii++) {
        // std::printf("iter %d \n", ii);
        // this operation does pointwise v1*v2 operation and stores the result in v1
        stat = cublasDdgmm(handle0, mode, m, n, (double *)dev_data1, lda, (double *)dev_data2, incx, (double *)dev_data1, ldc);
        // std::cout << (int)stat << std::endl;
        // if (stat != CUBLAS_STATUS_SUCCESS) std::printf("ERROR \n");
        // compute 1/v2
        RvectorReciprocal<<<blocksPerGrid, threadsPerBlock>>>((double *)dev_data2, (double *)scratch, numElements);
        //divide back
        stat = cublasDdgmm(handle0, mode, m, n, (double *)dev_data1, lda, (double *)scratch, incx, (double *)dev_data1, ldc);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::printf("CUBLAS elapsed time (in s): %.5f \t Approx time per multiply (in ms): %.5f \n",milliseconds/1000, milliseconds/Niter);
    finish_cublas();
    CUDA_RT_CALL(cudaFree(scratch));


    //////////////////////////////////////////////////////////////
    // same operation as before but with custom kernels
    //////////////////////////////////////////////////////////////

    cudaEventRecord(start);
    for (int ii = 0; ii < Niter; ii++) {
        // this operation does pointwise v1*v2 operation and stores the result in v1
        RRvectorMultiply<<<blocksPerGrid, threadsPerBlock>>>((double *)dev_data1, (double *)dev_data2, (double *)dev_data1, (double) 1.0, numElements);
        RRvectorDivide<<<blocksPerGrid, threadsPerBlock>>>((double *)dev_data1, (double *)dev_data2, (double *)dev_data1, (double) 1.0, numElements);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::printf("custom kernels elapsed time (in s): %.5f \t Approx time per multiply (in ms): %.5f \n",milliseconds/1000, milliseconds/Niter);



    // Copy output data to CPU
    CUDA_RT_CALL(cudaMemcpy(cpu_r_data1.data(), dev_data1, sizeof(scalar_type) * 2 * grid.NTOTAL_COMPLEX, cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaMemcpy(cpu_r_data2.data(), dev_data2, sizeof(scalar_type) * 2 * grid.NTOTAL_COMPLEX, cudaMemcpyDeviceToHost));


    std::printf("Output array:\n");
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++){
            for (int k = 0; k < 2; k++){
                idx = k + (grid.NZ/2+1)*2 * ( j + i * grid.NY);
                std::printf("v1[%d]= %f \t v2[%d]= %f \n", idx, cpu_r_data1[idx], idx, cpu_r_data2[idx]);
            }
        }
    }
    std::printf("=====\n");

    CUDA_RT_CALL(cudaFree(dev_data1));
    CUDA_RT_CALL(cudaFree(dev_data2));

    // finish_cufft();


    // std::printf("kmax: %.5f \n",wavevector.kmax);
    // return EXIT_SUCCESS;

};



void test_axpy() {
    // init_plan(grid.FFT_SIZE);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // cudaError_t devsyncherr;

    // Do forward and inverse transform
    int Niter=1;

    cpudata_t cpu_r_data1((size_t) 2*grid.NTOTAL_COMPLEX);
    cpudata_t cpu_r_data2((size_t) 2*grid.NTOTAL_COMPLEX);


    unsigned int idx;
    for (int i = 0; i < grid.NX; i++){
        for (int j = 0; j < grid.NY; j++){
            for (int k = 0; k < grid.NZ; k++){
                idx = k + (grid.NZ/2+1)*2 * ( j + i * grid.NY);
                cpu_r_data1[idx] = idx;
                cpu_r_data2[idx] = 2.0*idx;
            }
        }
    }
    for (int i = 0; i < grid.NX; i++){
        for (int j = 0; j < grid.NY; j++){
            for (int k = grid.NZ; k < grid.NZ+2; k++){
                idx = k + (grid.NZ/2+1)*2 * ( j + i * grid.NY);
                cpu_r_data1[idx] = 0.0;
                cpu_r_data2[idx] = 0.0;
            }
        }
    }
    cpu_r_data1[0] = 1.0;
    cpu_r_data2[0] = 1.0;

    std::printf("Input array:\n");
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++){
            for (int k = 0; k < 2; k++){
                idx = k + (grid.NZ/2+1)*2 * ( j + i * grid.NY);
                std::printf("v1[%d]= %f \t v2[%d]= %f \n", idx, cpu_r_data1[idx], idx, cpu_r_data2[idx]);
                // std::printf("v2[%d] %f \n", idx, cpu_r_data2[idx]);
            }
        }
    }
    std::printf("=====\n");

    // Create device data arrays
    scalar_type *dev_data1;
    scalar_type *dev_data2;
    CUDA_RT_CALL(cudaMalloc(&dev_data1, (size_t) sizeof(scalar_type) * 2 * grid.NTOTAL_COMPLEX));
    CUDA_RT_CALL(cudaMalloc(&dev_data2, (size_t) sizeof(scalar_type) * 2 * grid.NTOTAL_COMPLEX));
    std::printf("array size (in MiB): %f \n",(float) (sizeof(scalar_type) * 2 * grid.NTOTAL_COMPLEX/1e6));
    scalar_type *scratch;
    CUDA_RT_CALL(cudaMalloc(&scratch, (size_t) sizeof(scalar_type) * 2 * grid.NTOTAL_COMPLEX));

    // Copy input data to GPUs
    CUDA_RT_CALL(cudaMemcpy(dev_data1, cpu_r_data1.data(), (size_t) sizeof(scalar_type) * 2 * grid.NTOTAL_COMPLEX, cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(dev_data2, cpu_r_data2.data(), (size_t) sizeof(scalar_type) * 2 * grid.NTOTAL_COMPLEX, cudaMemcpyHostToDevice));

    float milliseconds = 0;
    int numElements = 2*grid.NTOTAL_COMPLEX;
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    scalar_type scale = 1.0;

    //////////////////////////////////////////////////////////////
    // kernels with double vectors
    //////////////////////////////////////////////////////////////

    cudaEventRecord(start);
    for (int ii = 0; ii < Niter; ii++) {
        // using doubles
        axpyDouble<<<blocksPerGrid, threadsPerBlock>>>( (double *)dev_data1, (double *)dev_data2, (double *)scratch, scale, (scalar_type) 1.0, numElements);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::printf("double vectors elapsed time (in s): %.5f \t Approx time per multiply (in ms): %.5f \n",milliseconds/1000, milliseconds/Niter);

    //////////////////////////////////////////////////////////////
    // kernels with complex vectors
    //////////////////////////////////////////////////////////////

    cudaEventRecord(start);
    for (int ii = 0; ii < Niter; ii++) {
        // using complex
        axpyComplex<<<blocksPerGrid, threadsPerBlock>>>( (data_type *)dev_data1, (data_type *)dev_data2, (data_type *)scratch, scale, (scalar_type) 1.0, grid.NTOTAL_COMPLEX);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::printf("complex vectors elapsed time (in s): %.5f \t Approx time per multiply (in ms): %.5f \n",milliseconds/1000, milliseconds/Niter);

    // Copy output data to CPU
    CUDA_RT_CALL(cudaMemcpy(cpu_r_data1.data(), dev_data1, sizeof(scalar_type) * 2 * grid.NTOTAL_COMPLEX, cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaMemcpy(cpu_r_data2.data(), dev_data2, sizeof(scalar_type) * 2 * grid.NTOTAL_COMPLEX, cudaMemcpyDeviceToHost));


    std::printf("Output array:\n");
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++){
            for (int k = 0; k < 2; k++){
                idx = k + (grid.NZ/2+1)*2 * ( j + i * grid.NY);
                std::printf("v1[%d]= %f \t v2[%d]= %f \n", idx, cpu_r_data1[idx], idx, cpu_r_data2[idx]);
            }
        }
    }
    std::printf("=====\n");

    CUDA_RT_CALL(cudaFree(dev_data1));
    CUDA_RT_CALL(cudaFree(dev_data2));

    // finish_cufft();


    // std::printf("kmax: %.5f \n",wavevector.kmax);
    // return EXIT_SUCCESS;

};*/
