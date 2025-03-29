import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
import h5py
import glob

sp_savedir = "./"
sp_savename = 'snap'

sp_data_list = glob.glob(sp_savedir+'*.h5')

nx = 64
ny = 64
nz = 64

lx = 1.0
ly = 1.0
lz = 1.0

nu = 1./200.

x = -0.5 + lx * (np.arange(nx)) / nx
y = -0.5 + ly * (np.arange(ny)) / ny
z = -0.5 + lz * (np.arange(nz)) / nz

X, Y, Z = np.meshgrid(x,y,z,indexing='ij')


vx_0 =   np.zeros(X.shape)
vy_0 =   np.sin(2.0*np.pi*Y/ly) * np.cos(2.0*np.pi*Z/lz)
vz_0 = - np.cos(2.0*np.pi*Y/ly) * np.sin(2.0*np.pi*Z/lz)

tol = 1e-15

# with h5py.File(sp_savedir+'{:s}{:04d}.h5'.format(sp_savename,0), 'w') as data_0:
data_0 = h5py.File(sp_savedir+'{:s}{:04d}.h5'.format(sp_savename,0), 'w')
dset = data_0.create_dataset("step", (1,), dtype='i4')
dset[0] = 0

dset = data_0.create_dataset("t_end", (1,), dtype='f8')
dset[0] = 10.0

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

data_0.close()
