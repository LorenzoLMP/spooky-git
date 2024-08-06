import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
import h5py
import glob
import os

sp_savedir = "../../build/tests/mti-eigenmode/data/"
sp_savename = 'snap'

if not os.path.exists(sp_savedir+'Plots'):
    os.mkdir(sp_savedir+'Plots')


sp_data_list = glob.glob(sp_savedir+'*.h5')

# if (len(py_data_list) != len(sp_data_list)):
#     print('Number of datafiles not the same!')

nx = 1024
ny = 4
nz = 4

lx = 1.0
ly = 1.0
lz = 1.0

reynolds_m = 100000
reynolds_ani = 100
sigma  = 0.9341413811120219
Pe     = reynolds_ani
Reeta  = reynolds_m
kparallel  = (2.0*np.pi/lx)*12.0
B0 = 1e-4
N2 = 0.1

x = -0.5 + lx * (np.arange(nx)) / nx
y = -0.5 + ly * (np.arange(ny)) / ny
z = -0.5 + lz * (np.arange(nz)) / nz

X, Y, Z = np.meshgrid(x,y,z,indexing='ij')

vx_0 = np.zeros(X.shape)
vy_0 = np.zeros(X.shape)
vz_0 = -0.00001*np.sin(kparallel*X)

bx_0 = B0*np.ones(X.shape)
by_0 = np.zeros(X.shape)
bz_0 = -0.00001*np.cos(kparallel*X)*B0*kparallel/(sigma+kparallel*kparallel/Reeta)

th_0 = 1.0/(sigma + kparallel*kparallel/Pe)*(N2 - kparallel*kparallel/Pe/(sigma+kparallel*kparallel/Reeta) ) * vz_0

tol = 1e-15


# e.g. t = 5
# vx_0[0:10,0,0]*np.exp(-2.0*nu*t*(2.0*np.pi)**2)
# vy_0[0:10,10,0]*np.exp(-2.0*nu*5*(2.0*np.pi)**2)

for i in range(len(sp_data_list)):
# for i in range(0,1):
# for i in range(3,4):
    data_sp = h5py.File(sp_savedir+'{:s}{:04d}.h5'.format(sp_savename,i), 'r')
    vx = np.reshape(data_sp['vx'],(nx,ny,nz))
    vy = np.reshape(data_sp['vy'],(nx,ny,nz))
    vz = np.reshape(data_sp['vz'],(nx,ny,nz))

    bx = np.reshape(data_sp['bx'],(nx,ny,nz))
    by = np.reshape(data_sp['by'],(nx,ny,nz))
    bz = np.reshape(data_sp['bz'],(nx,ny,nz))

    th = np.reshape(data_sp['th'],(nx,ny,nz))

    t =  data_sp['t_save'][0]
    data_sp.close()

    vx_analytical = vx_0
    vy_analytical = vy_0
    vz_analytical = vz_0 * np.exp(sigma*t)

    bx_analytical = bx_0
    by_analytical = by_0
    bz_analytical = bz_0 * np.exp(sigma*t)

    th_analytical = th_0 * np.exp(sigma*t)

    fig, ax = plt.subplots(3,1, figsize=(9,10))

    ax[0].plot(x, vz_analytical[:,0,0], ls='', marker='x', markerfacecolor='none', label='analytical')
    ax[0].plot(x, vz[:,0,0], ls='', marker='s', markerfacecolor='none', label='spooky')

    ax[0].set_title(r'$\delta v_z$')
    ax[0].legend()

    ax[1].plot(x, bz_analytical[:,0,0], ls='', marker='x', markerfacecolor='none', label='analytical')
    ax[1].plot(x, bz[:,0,0], ls='', marker='s', markerfacecolor='none', label='spooky')

    ax[1].set_title(r'$\delta B_z$')
    ax[1].legend()

    ax[2].plot(x, th_analytical[:,0,0], ls='', marker='x', markerfacecolor='none', label='analytical')
    ax[2].plot(x, th[:,0,0], ls='', marker='s', markerfacecolor='none', label='spooky')

    ax[2].set_title(r'$\theta$')
    ax[2].legend()

    fig.tight_layout(pad=1.00, h_pad=None, w_pad=None, rect=None)

    plt.savefig(sp_savedir+'Plots/'+'{:s}{:04d}.pdf'.format(sp_savename,i))

    # plt.show()
    plt.close()

    # L2_err = np.sum(np.power(vx-vx_analytical,2.0))
    # L2_err += np.sum(np.power(vy-vy_analytical,2.0))
    # L2_err += np.sum(np.power(vz-vz_analytical,2.0))
    #
    # print('t = {:10.4f} \t L2 error = {:0.2e}'.format(t,L2_err))

    # if (L2_err < tol):
    #     print('i = %d \t L2 error = %.2e ... PASSED'%(i,L2_err))
    # else:
    #     print('i = %d \t L2 error = %.2e ... NOT PASSED'%(i,L2_err))
