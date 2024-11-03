#include <cuda_runtime.h>
#include <cufftXt.h>
// #include "spooky.hpp"
#include "cufft_utils.h"
#include "define_types.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/complex.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>
