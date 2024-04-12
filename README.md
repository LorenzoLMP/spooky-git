# SPOOKY
                 ____________
               --            --
             /                  \\
            /                    \\
           /     __               \\
          |     /  \       __      ||
          |    |    |     /  \     ||
                \__/      \__/
         |             ^            ||
         |                          ||
         |                          ||
        |                            ||
        |                            ||
        |                            ||
         \__         ______       __//
            \       //     \_____//
             \_____//


## Description

Pseudospectral code to do HD/MHD simulations on a triply-periodic box on the GPU with CUDA. Work in progress.

## Prerequisites 

The current implementation of SPOOKY requires:

1. a CUDA compiler (tested with cuda-11.8 and cuda-12.0)
2. CUDA toolkit
3. cmake (minimum 3.24)
4. Python 3.+ with numpy, matplotlib, argparse (necessary for some tests)
5. `libconfig` and `HDF5` libraries (can be installed automatically if not present)

## Installation

```
git clone git@github.com:LorenzoLMP/spooky-git.git
cd spooky-git
```

## Compiling with cmake (instructions to compile and run on newton cluster to follow)

Create build directory if not already present for out-of-source build (recommended)

```
$ mkdir build
$ cd build
```

A typical build command looks like this:

```
$ cmake -DBUILD_TESTS=ON -DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc -DHDF5_ROOT=/path/to/hdf5/ -DLIBCONFIG_ROOT=/path/to/libconfig/ -DCMAKE_CUDA_ARCHITECTURES="XX" ..
```

1. The cuda architectures have to be chosen based on the hardware that is available. 75 for NVIDIA Quadro RTX 8000, 80 for A100.
2. Depending on the version of your default g++ compiler, it might be incompatible with the .... If so, add the option ```-DCMAKE_CUDA_FLAGS="-ccbin /path/to/g++"``` with the path to a compatible version of g++
3. If you don't want to build the tests, simply do ```-DBUILD_TESTS=OFF``` or omit.
4. If you don't have libconfig or hdf5 installed, omit the option ```-DLIBCONFIG_ROOT``` or ```-DHDF5_ROOT``` and CMake will attempt to automatically donwload and build the appropriate version of the libraries.

If the configuration step was successful, now simply compile as:

```
$ make clean && make -j 8
```

The SPOOKY executable can be run as
```
$ ./src/spooky --input-dir /path/to/input/dir
```

## Running tests

If you want to run the tests (```-DBUILD_TESTS=ON```) do instead:

```
$ ctest -V -R "spooky"
```

which will run all the spooky tests and show the output.

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


<!--
## CUDA APIs involved
- [cufftExecC2C API](https://docs.nvidia.com/cuda/cufft/index.html#function-cufftexecc2c-cufftexecz2z)

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
