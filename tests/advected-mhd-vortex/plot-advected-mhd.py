import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
import h5py
import glob
import os
from matplotlib.colors import LogNorm, Normalize

sp_savedir = "../../build/tests/advected-mhd-vortex/data/"
sp_savename = 'snap'

if not os.path.exists(sp_savedir+'Plots'):
    os.mkdir(sp_savedir+'Plots')


sp_data_list = glob.glob(sp_savedir+'*.h5')

# if (len(py_data_list) != len(sp_data_list)):
#     print('Number of datafiles not the same!')

nx = 64
ny = 64
nz = 64

lx = 1.0
ly = 1.0
lz = 1.0

nu = 1./10000.

v0 = 0.1
u0 = 0.1

x = -0.5 + lx * (np.arange(nx)) / nx
y = -0.5 + ly * (np.arange(ny)) / ny
z = -0.5 + lz * (np.arange(nz)) / nz

X, Y, Z = np.meshgrid(x,y,z,indexing='ij')

def vel_analytical(X,Y,Z,t):
    Xt = X - u0*t
    Yt = Y - v0*t
    R2t = Xt**2 + Yt**2
    vx =   v0 - 1./(2.0 * np.pi) * np.exp( (1.0  - 100.*R2t)/2.0 ) * Yt
    vy =   u0 + 1./(2.0 * np.pi) * np.exp( (1.0  - 100.*R2t)/2.0 ) * Xt
    vz =   np.zeros(X.shape)
    return vx, vy, vz

def mag_analytical(X,Y,Z,t):
    Xt = X - u0*t
    Yt = Y - v0*t
    R2t = Xt**2 + Yt**2
    Bx =   - 1./(2.0 * np.pi) * np.exp( (1.0  - 100.*R2t)/2.0 ) * Yt
    By =     1./(2.0 * np.pi) * np.exp( (1.0  - 100.*R2t)/2.0 ) * Xt
    Bz =   np.zeros(X.shape)
    return Bx, By, Bz

vx_0, vy_0, vz_0 = vel_analytical(X,Y,Z,0.0)
bx_0, by_0, bz_0 = mag_analytical(X,Y,Z,0.0)

tol = 1e-15
# omega_A = (2.0*np.pi/lz)

# e.g. t = 5
# vx_0[0:10,0,0]*np.exp(-2.0*nu*t*(2.0*np.pi)**2)
# vy_0[0:10,10,0]*np.exp(-2.0*nu*5*(2.0*np.pi)**2)

vmin = -1./(2.0 * np.pi)/10
vmax =  1./(2.0 * np.pi)/10
norm = Normalize(vmin=vmin,vmax=vmax)
extent = [-0.5,0.5,-0.5,0.5]

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

    t =  data_sp['t_save'][0]
    data_sp.close()

    vx_analytical, vy_analytical, vz_analytical = vel_analytical(X,Y,Z,t)
    bx_analytical, by_analytical, bz_analytical = mag_analytical(X,Y,Z,t)

    fig, ax = plt.subplots(1,2)

    im0 = ax[0].imshow(vx_analytical[:,:,0]-v0, origin='lower', norm=norm,cmap='RdBu_r',extent=extent)
    im1 = ax[1].imshow(vx[:,:,0]-v0, origin='lower', norm=norm,cmap='RdBu_r',extent=extent)

    ax[0].set_title(r'analytical')
    ax[1].set_title(r'spooky')

    cbar = fig.colorbar(im0, ax=ax[0], orientation='horizontal')
    cbar.set_label(r'$\delta v_x$')

    cbar = fig.colorbar(im1, ax=ax[1], orientation='horizontal')
    cbar.set_label(r'$\delta v_x$')
    # ax[0].legend()

    # ax[1].plot(z, by_analytical[0,0,:], ls='', marker='x', markerfacecolor='none', label='analytical')
    # ax[1].plot(z, by[0,0,:], ls='', marker='s', markerfacecolor='none', label='spooky')
    #
    # ax[1].set_title(r'$\delta B_y$')
    # ax[1].legend()

    fig.suptitle(r'$v_x$')

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
