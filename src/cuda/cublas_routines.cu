#include <cublas_v2.h>
#include "cublas_routines.hpp"

cublasHandle_t handle0;
cublasHandle_t handle1;

void init_cublas(){
    cublasCreate(&handle0);
    cublasCreate(&handle1);
}

void finish_cublas(){
    cublasDestroy(handle0);
    cublasDestroy(handle1);
}
