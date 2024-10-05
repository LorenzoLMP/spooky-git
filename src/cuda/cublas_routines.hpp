#include "define_types.hpp"
#include <cublas_v2.h>

extern cublasHandle_t handle0;
extern cublasHandle_t handle1;
// extern cublasStatus_t stat;

void init_cublas();
void finish_cublas();
