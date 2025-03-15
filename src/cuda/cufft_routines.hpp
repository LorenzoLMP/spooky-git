// #include <cufftXt.h>
#include "common.hpp"

#include <cuda_runtime.h>
#include "cufft_utils.h"

// extern const int threadsPerBlock;
class Supervisor;

void r2c_fft(void *r_data_in, void *c_data_out);
void c2r_fft(void *c_data_in, void *r_data_out);

void r2c_fft(void *r_data_in, void *c_data_out, Supervisor *supervisor);
void c2r_fft(void *c_data_in, void *r_data_out, Supervisor *supervisor);

void init_plan(const size_t *fft_size);
void finish_cufft();

void Complex2RealFields(data_type* ComplexField_in, int num_fields);
void Complex2RealFields(data_type* ComplexField_in, scalar_type* RealField_out, int num_fields);
