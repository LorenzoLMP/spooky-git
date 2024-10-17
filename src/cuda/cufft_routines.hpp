// #include <cufftXt.h>
// #include "spooky.hpp"
#include "define_types.hpp"

#include <cuda_runtime.h>
#include "cufft_utils.h"

// extern const int threadsPerBlock;
class Supervisor;

void r2c_fft(void *r_data_in, void *c_data_out);
void r2c_fft(void *r_data_in, void *c_data_out, Supervisor *supervisor);

void c2r_fft(void *c_data_in, void *r_data_out);
void c2r_fft(void *c_data_in, void *r_data_out, Supervisor *supervisor);

void init_plan(const size_t *fft_size);
void finish_cufft();
