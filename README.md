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

Pseudospectral code to do HD/MHD simulations on a triply-periodic box on (one) NVidia GPU with CUDA. Largely inspired by the Snoopy code (https://ipag.osug.fr/~lesurg/snoopy). Work in progress!

## Prerequisites 

The current implementation of SPOOKY requires:

1. a CUDA compiler (tested with cuda-11.8 and cuda-12.0)
2. CUDA toolkit
3. cmake (minimum 3.24)
4. Python 3.+ with numpy, matplotlib, argparse (necessary for some tests)
5. `libconfig` and `HDF5` libraries (can be installed automatically if not present)

# Installation

```bash
git clone git@github.com:LorenzoLMP/spooky-git.git
cd spooky-git
```

The installation can be done either directly on the host or inside a container (see below).

## Compile with CMake

Create build directory if not already present for out-of-source build (recommended)

```bash
$ mkdir build
$ cd build
```

A typical build command looks like this:

```bash
$ cmake -DBUILD_TESTS=ON -DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc -DHDF5_ROOT=/path/to/hdf5/ -DLIBCONFIG_ROOT=/path/to/libconfig/ -DCMAKE_CUDA_ARCHITECTURES="XX" ..
```

1. The cuda architectures have to be chosen based on the hardware that is available. 75 for NVIDIA Quadro RTX 8000, 80 for A100.
2. Depending on the version of your default g++ compiler, it might be incompatible with the .... If so, add the option ```-DCMAKE_CUDA_FLAGS="-ccbin /path/to/g++"``` with the path to a compatible version of g++
3. If you don't want to build the tests, simply do ```-DBUILD_TESTS=OFF``` or omit.
4. If you don't have libconfig or hdf5 installed, omit the option ```-DLIBCONFIG_ROOT``` or ```-DHDF5_ROOT``` and CMake will attempt to automatically donwload and build the appropriate version of the libraries.

If the configuration step was successful, now simply compile as:

```bash
$ make clean && make -j 8
```

The SPOOKY executable can be run as
```bash
$ ./src/spooky --input-dir /path/to/input/dir
```


## Using a container

Users can choose to use the definition file spooky-container.def inside the repository to create a container that includes the NVIDIA CUDA libraries to run and develop spooky. This option can be useful for users on a shared HPC resources.

Here below are the steps with Apptainer (which is assumed to be installed on the system). Apptainer supports running GPU applications from withing the container. The host has to have a driver and matching library installation of CUDA. Please refer to the [GPU Support page](https://apptainer.org/docs/user/main/gpu.html) of Apptainer for further information.


Create a temporary directory in which to run the container:
```bash
$ export APPTAINER_TMPDIR=/path/to/temp/dir
$ mkdir -p $APPTAINER_TMPDIR && cd $APPTAINER_TMPDIR
```

Clone spooky-git:
```bash
$ git clone https://github.com/LorenzoLMP/spooky-git.git
```

Build the .sif image:
```bash
$ apptainer build  spooky-container.sif spooky-git/spooky-container.def
```

Assuming the build process has been successful, you can now open a terminal in the container using the --nv flag which binds the cuda libraries of the host:
```bash
$ apptainer shell --nv  spooky-container.sif
```
From this point onwards the instructions to compile and run as the same:

```bash
$ cd spooky-git
$ mkdir build
$ cd build
```

Now the cmake command is simply:

```bash
$ cmake -DBUILD_TESTS=ON  -DCMAKE_CXX_FLAGS="-O3 -std=c++2a" 
```


# Running tests

If you want to run the tests (```-DBUILD_TESTS=ON```) do instead: (NOTE: to verify the sts scalings the code will use a Forward Euler instead of the RK3, and a custom timestep)

```bash
$ ctest -V -R "spooky" -E "sts"
```

which will run all the spooky tests (excluding the sts suite) and show the output.


# Configurations

## On local laptop (last update: 2025-01-05)

```bash
cmake -DBUILD_TESTS=ON ..
```

## On Newton (last update: 2026-03-01)

For interactive jobs:
```bash
srun -p a100 --gres=gpu:1 --job-name "GpuInteractiveJob" --time=04:00:00 --pty bash
```

```bash
$ source load_modules
```

```bash
cd build
rm -rf *
```

```bash
$ cmake -DBUILD_TESTS=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DCMAKE_CUDA_FLAGS="" -DCMAKE_CXX_FLAGS="-O3 -std=c++2a" -DHDF5_ROOT=/home/lperrone/myhdf5/hdf5/ -DLIBCONFIG_ROOT=/home/lperrone/mylibconfig/libconfig/ -DCMAKE_CUDA_ARCHITECTURES="80" ..

```


```bash
$ make clean && make -j 8
```

or just for one executable:

```bash
$ make clean && make spooky -j 8
```

For the generic test problem:

```bash
./problems/generic/spooky --input-dir ../problems/generic/ --output-dir /lustre/lperrone/spooky/tests/tmp --stats 100

```

You can also submit a job using slurm as follows:

```
#!/bin/bash
#! Which partition (queue) should be used
#SBATCH -p a100
#SBATCH -J SPOOKY_job
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

### compute nodes
#SBATCH --nodes=1
###  MPI ranks
#SBATCH --ntasks=1
###  MPI ranks per node
#SBATCH --ntasks-per-node=1
###  tasks per MPI rank(eg OMP tasks)
#SBATCH --cpus-per-task=1
###  gpu per node
#SBATCH --gres=gpu:1

#!How much wallclock time will be required (HH:MM:SS)
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lperrone@aip.de
##SBATCH --begin=now+16hours

source /home/lperrone/spooky-git/load_modules

./spooky-mti3d --input-dir ./ --output-dir /lustre/lperrone/spooky/tests/Pm4_beta5e5T_Re_6400_Rm_12800_Pe_1600_N2_2e-1_new --stats 1000
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
