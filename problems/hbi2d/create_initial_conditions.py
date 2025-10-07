import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
import h5py
import glob
import argparse

sp_savedir = "/home/lorenzolmp/Documents/cuda-code/spooky-data/hbi2d/"
sp_savename = 'snap'

sp_data_list = glob.glob(sp_savedir+'*.h5')

rng = np.random.default_rng(seed=98765)

nx = 512
ny = 512
nz = 2

lx = 1.0
ly = 1.0
lz = 1.0


x = -0.5 + lx * (np.arange(nx)) / nx
y = -0.5 + ly * (np.arange(ny)) / ny
z = -0.5 + lz * (np.arange(nz)) / nz

X, Y, Z = np.meshgrid(x,y,z,indexing='ij')

kx_vec = 2.0*np.pi*np.fft.fftfreq(nx, d=lx/nx)
ky_vec = 2.0*np.pi*np.fft.fftfreq(ny, d=ly/ny)
kz_vec = 2.0*np.pi*np.fft.rfftfreq(nz, d=lz/nz)

KX, KY, KZ = np.meshgrid(kx_vec, ky_vec, kz_vec, sparse=False, indexing='ij')

def rand(size):
    return rng.uniform(low=-0.5, high=0.5, size=size)


fact = (27.0/8.0*nx*ny*nz)**0.5

vx_0_hat = 0.01*rand(KX.shape) * np.exp(1j*2.0*np.pi*rand(KX.shape)) * fact
vy_0_hat = 0.01*rand(KX.shape) * np.exp(1j*2.0*np.pi*rand(KX.shape)) * fact

vx_0 =   np.fft.irfftn(vx_0_hat)
vy_0 =   np.fft.irfftn(vy_0_hat)
vz_0 =   np.zeros(X.shape)

bx_0 =   np.zeros(X.shape)
by_0 =   1e-5*np.ones(X.shape)
bz_0 =   np.zeros(X.shape)

th_0 =   np.zeros(X.shape)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--output-dir',
                        default=sp_savedir,
                        # action='store_const',
                        help='full path to output dir')

    args = parser.parse_args()
    print("savedir is %s"%(args.output_dir))

    # with h5py.File(sp_savedir+'{:s}{:04d}.h5'.format(sp_savename,0), 'w') as data_0:
    data_0 = h5py.File(args.output_dir+'/data/'+'{:s}{:04d}.h5'.format(sp_savename,0), 'w')

    dset = data_0.create_dataset("step", (1,), dtype='i4')
    dset[0] = 0

    dset = data_0.create_dataset("t_lastsnap", (1,), dtype='f8')
    dset[0] = 0.0

    dset = data_0.create_dataset("t_lastvar", (1,), dtype='f8')
    dset[0] = 0.0

    dset = data_0.create_dataset("t_save", (1,), dtype='f8')
    dset[0] = 0.0

    dset = data_0.create_dataset("t_start", (1,), dtype='f8')
    dset[0] = 0.0

    dset = data_0.create_dataset("vx", (nx*ny*nz,), dtype='f8')
    dset[:] = vx_0.flatten()

    dset = data_0.create_dataset("vy", (nx*ny*nz,), dtype='f8')
    dset[:] = vy_0.flatten()

    dset = data_0.create_dataset("vz", (nx*ny*nz,), dtype='f8')
    dset[:] = vz_0.flatten()

    dset = data_0.create_dataset("bx", (nx*ny*nz,), dtype='f8')
    dset[:] = bx_0.flatten()

    dset = data_0.create_dataset("by", (nx*ny*nz,), dtype='f8')
    dset[:] = by_0.flatten()

    dset = data_0.create_dataset("bz", (nx*ny*nz,), dtype='f8')
    dset[:] = bz_0.flatten()

    dset = data_0.create_dataset("th", (nx*ny*nz,), dtype='f8')
    dset[:] = th_0.flatten()

    data_0.close()

if __name__ == '__main__':
    main()

