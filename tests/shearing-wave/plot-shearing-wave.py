import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
import h5py
import glob
import os
from matplotlib.colors import LogNorm, Normalize
import math

sp_savedir = "../../build/tests/shearing-wave/data/"
sp_savename = 'snap'

sp_data_list = glob.glob(sp_savedir+'*.h5')

if not os.path.exists(sp_savedir+'Plots'):
    os.mkdir(sp_savedir+'Plots')

# if (len(py_data_list) != len(sp_data_list)):
#     print('Number of datafiles not the same!')

nx = 128
ny = 192
nz = 16

lx = 1.0
ly = 1.5
lz = 1.0

shear = 1.5
omega = 1.0
q = shear/omega

nu = 1./2000.

x = -lx/2 + lx * (np.arange(nx)) / nx
y = -ly/2 + ly * (np.arange(ny)) / ny
z = -lz/2 + lz * (np.arange(nz)) / nz

X, Y, Z = np.meshgrid(x,y,z,indexing='ij')

kx_vec = 2.0*np.pi*np.fft.fftfreq(nx, d=lx/nx)
ky_vec = 2.0*np.pi*np.fft.fftfreq(ny, d=ly/ny)
kz_vec = 2.0*np.pi*np.fft.fftfreq(nz, d=lz/nz)

KX, KY, KZ = np.meshgrid(kx_vec, ky_vec, kz_vec, sparse=False, indexing='ij')

# vx_0 =   np.sin(2.0*np.pi*X/lx) * np.cos(2.0*np.pi*Y/ly)
# vy_0 = - np.cos(2.0*np.pi*X/lx) * np.sin(2.0*np.pi*Y/ly)
# vz_0 =   np.zeros(X.shape)
kx0 = 0.0
ky0 = 2.0*np.pi/ly
kz0 = 0.0
vx_0 =   np.zeros(X.shape)
vy_0 =   np.zeros(X.shape)
vz_0 =   np.cos(ky0*Y)

tol = 1e-15

vmin = -1.
vmax =  1.
extent = [-lx/2,lx/2,-ly/2,ly/2]

def shear_quantity(w, tremap):
    # the convention is that this function shears
    # the coords in the same direction of the
    # background flow. If one wants to use it to shear
    # "back" to the last t_remap then t_remap must be <0
    w_sheared = np.real(np.fft.ifft(np.fft.fft(w,axis=1)*np.exp(1j*(KY)*(X)*shear*tremap),axis=1))
    return w_sheared

for i in range(len(sp_data_list)):
# for i in range(1,2):
# for i in range(3,4):
    data_sp = h5py.File(sp_savedir+'{:s}{:04d}.h5'.format(sp_savename,i), 'r')
    vx = np.reshape(data_sp['vx'],(nx,ny,nz))
    vy = np.reshape(data_sp['vy'],(nx,ny,nz))
    vz = np.reshape(data_sp['vz'],(nx,ny,nz))
    # print(vy[0,0:5,0:5])
    # print(vz[0,0:5,0:5])
    t =  data_sp['t_save'][0]
    # tvel = math.fmod(t, 2.0 * ly / (shear * lx))

    data_sp.close()

    vx_analytical = vx_0
    vy_analytical = vy_0
    vz_analytical = shear_quantity(vz_0, t)*np.exp(-nu*ky0**2*t)

    # lecture 12 Gordon Ogilvie
    viscous_decay = np.exp(-nu*( (kx0**2 + ky0**2 + kz0**2)*t + shear*kx0*ky0*t**2 + (1./3.)*shear**2*ky0**2*t**3 ) )

    vz_analytical *= viscous_decay

    norm = Normalize(vmin=np.min(vz_analytical),vmax=np.max(vz_analytical))


    fig, ax = plt.subplots(1,2, sharey=True)

    im0 = ax[0].imshow(vz_analytical[:,:,0].T, origin='lower', norm=norm,cmap='RdBu_r',extent=extent, interpolation='none')
    im1 = ax[1].imshow(vz[:,:,0].T, origin='lower', norm=norm,cmap='RdBu_r',extent=extent, interpolation='none')

    ax[0].set_title(r'analytical')
    ax[1].set_title(r'spooky')

    cbar = fig.colorbar(im0, ax=ax[0], orientation='horizontal')
    cbar.set_label(r'$v_z$')

    cbar = fig.colorbar(im1, ax=ax[1], orientation='horizontal')
    cbar.set_label(r'$v_z$')
    # ax[0].legend()

    # ax[1].plot(z, by_analytical[0,0,:], ls='', marker='x', markerfacecolor='none', label='analytical')
    # ax[1].plot(z, by[0,0,:], ls='', marker='s', markerfacecolor='none', label='spooky')
    #
    ax[0].set_xlabel(r'$x$')
    ax[1].set_xlabel(r'$x$')
    ax[1].set_ylabel(r'$y$')
    # ax[1].legend()

    fig.suptitle(r'$v_z$')

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
