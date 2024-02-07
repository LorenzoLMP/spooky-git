# SPOOKY


## Instructions for compiling with cmake

create build directory if not already present

```
$ mkdir build
$ cd build
$ cmake ..
$ make clean && make -j 8
$ ./src/spooky --input-dir /path/to/input/dir
```

## Running tests

If you want to run the tests do instead:
```
$ conda activate astro-vtk (for python test scripts)
$ cmake -DBUILD_TESTS=ON ..
$ make clean && make -j 8
$ ctest
```
## Steps for profiling
```
$ nsys start --stop-on-exit=false
$ nsys launch --trace=cuda,nvtx spooky
$ nsys stop

$ sudo -E /opt/nvidia/hpc_sdk/Linux_x86_64/23.1/profilers/Nsight_Systems/bin/nsys-ui &
```
File -> Open .nsys-rep
```
$ sudo /opt/nvidia/hpc_sdk/Linux_x86_64/23.1/profilers/Nsight_Compute/ncu --target-processes all spooky
```
for a single kernel
```
$ sudo /opt/nvidia/hpc_sdk/Linux_x86_64/23.1/profilers/Nsight_Compute/ncu --export "/home/lorenzolmp/Documents/NVIDIA Nsight Compute/report%i" --force-overwrite --target-processes all --kernel-name axpyDouble --launch-count 1 spooky
```
## Description

Pseudospectral code to do HD/MHD simulations on a triply-periodic box on the GPU with CUDA. Work in progress.

## CUDA APIs involved
- [cufftExecC2C API](https://docs.nvidia.com/cuda/cufft/index.html#function-cufftexecc2c-cufftexecz2z)


<!--
## Building (make)

## Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- [CMake](https://cmake.org/download) version 3.18 minimum

## Build command on Linux
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```
Make sure that CMake finds expected CUDA Toolkit. If that is not the case you can add argument `-DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc` to cmake command.

# Usage 1
TBC-->
