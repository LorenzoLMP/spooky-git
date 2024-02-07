import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
import h5py
import glob

sp_savedir = "./data/"
sp_savename = 'snap'

sp_data_list = glob.glob(sp_savedir+'*.h5')

# if (len(py_data_list) != len(sp_data_list)):
#     print('Number of datafiles not the same!')

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

# vx_0 =   np.sin(2.0*np.pi*X/lx) * np.cos(2.0*np.pi*Y/ly)
# vy_0 = - np.cos(2.0*np.pi*X/lx) * np.sin(2.0*np.pi*Y/ly)
# vz_0 =   np.zeros(X.shape)

vx_0 =   np.zeros(X.shape)
vy_0 =   np.sin(2.0*np.pi*Y/ly) * np.cos(2.0*np.pi*Z/lz)
vz_0 = - np.cos(2.0*np.pi*Y/ly) * np.sin(2.0*np.pi*Z/lz)

tol = 1e-15


# e.g. t = 5
# vx_0[0:10,0,0]*np.exp(-2.0*nu*t*(2.0*np.pi)**2)
# vy_0[0:10,10,0]*np.exp(-2.0*nu*5*(2.0*np.pi)**2)

# for i in range(len(sp_data_list)):
for i in range(1,2):
# for i in range(3,4):
    data_sp = h5py.File(sp_savedir+'{:s}{:04d}.h5'.format(sp_savename,i), 'r')
    vx = np.reshape(data_sp['vx'],(nx,ny,nz))
    vy = np.reshape(data_sp['vy'],(nx,ny,nz))
    vz = np.reshape(data_sp['vz'],(nx,ny,nz))
    print(vy[0,0:5,0:5])
    print(vz[0,0:5,0:5])
    t =  data_sp['t_save'][0]
    data_sp.close()

    vx_analytical = vx_0 * np.exp(-2.0*nu*t*(2.0*np.pi)**2)
    vy_analytical = vy_0 * np.exp(-2.0*nu*t*(2.0*np.pi)**2)
    vz_analytical = vz_0 * np.exp(-2.0*nu*t*(2.0*np.pi)**2)

    L2_err = np.sum(np.power(vx-vx_analytical,2.0))
    L2_err += np.sum(np.power(vy-vy_analytical,2.0))
    L2_err += np.sum(np.power(vz-vz_analytical,2.0))

    print('t = {:10.4f} \t L2 error = {:0.2e}'.format(t,L2_err))

    # if (L2_err < tol):
    #     print('i = %d \t L2 error = %.2e ... PASSED'%(i,L2_err))
    # else:
    #     print('i = %d \t L2 error = %.2e ... NOT PASSED'%(i,L2_err))
