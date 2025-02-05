import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
import h5py
import glob
import os
from matplotlib.colors import LogNorm, Normalize
import math

sp_savedir = "../../build/tests/mri-eigenmode/data/"
sp_savename = 'snap'

sp_data_list = glob.glob(sp_savedir+'*.h5')

if not os.path.exists(sp_savedir+'Plots'):
    os.mkdir(sp_savedir+'Plots')

# if (len(py_data_list) != len(sp_data_list)):
#     print('Number of datafiles not the same!')

nx = 64
ny = 64
nz = 64

lx = 1.0
ly = 1.5
lz = 1.0

shear = 1.5
omega = 1.0
q = shear/omega

reynolds = 3000
reynolds_m = 3000

# Axisymmetric MRI eigenmode with k_x, k_z, Pm=1
# and vertical background magnetic field


B0z = 0.1
kZ     = 2.0*2.0*np.pi/lz
kX     = 1.0*2.0*np.pi/lx
gamma2 = kZ*kZ/(kZ*kZ + kX*kX)
kappa2 = 2*omega*(2*omega - shear)
omegaA2    = B0z*B0z*kZ*kZ
Omega = omega
S = shear
Omega2 = Omega*Omega

# solve for omega_nu2, with omega_nu = -i * sigma_nu
omega_nu2 = omegaA2 + 0.5*kappa2*gamma2*( 1. - np.sqrt(1. + 16.*omegaA2*Omega2/(kappa2*kappa2*gamma2) ) )
sigma_nu = np.sqrt(-omega_nu2)
sigma = sigma_nu - (kX*kX + kZ*kZ)/reynolds
# sigma_nu   = sigma + (kX*kX + kZ*kZ)/reynolds
# we assume sigma_nu and sigma_eta are equal
sigma_eta  = sigma + (kX*kX + kZ*kZ)/reynolds_m


x = -lx/2 + lx * (np.arange(nx)) / nx
y = -ly/2 + ly * (np.arange(ny)) / ny
z = -lz/2 + lz * (np.arange(nz)) / nz

X, Y, Z = np.meshgrid(x,y,z,indexing='ij')

kx_vec = 2.0*np.pi*np.fft.fftfreq(nx, d=lx/nx)
ky_vec = 2.0*np.pi*np.fft.fftfreq(ny, d=ly/ny)
kz_vec = 2.0*np.pi*np.fft.fftfreq(nz, d=lz/nz)

KX, KY, KZ = np.meshgrid(kx_vec, ky_vec, kz_vec, sparse=False, indexing='ij')

a = 1e-4

vx_0 =   a*np.cos(kX*X + kZ*Z)
vy_0 =   vx_0 * (S*omegaA2/(sigma_eta*sigma_eta) - (2*Omega - S))/(sigma_nu + omegaA2/sigma_eta)
vz_0 =   -(kX/kZ)*a*np.cos(kX*X + kZ*Z)

bx_0 = -a*np.sin(kX*X + kZ*Z)*B0z*kZ/sigma_eta
by_0 =  bx_0 * ( - S/(sigma_eta) + (S*omegaA2/(sigma_eta*sigma_eta)   - (2*Omega - S)) /(sigma_nu + omegaA2/sigma_eta) )
bz_0 = B0z * (1.0 + (kX/kZ)*a*np.sin(kX*X + kZ*Z)*kZ/sigma_eta )

tol = 1e-15

# vmin = -1.
# vmax =  1.
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

    bx = np.reshape(data_sp['bx'],(nx,ny,nz))
    by = np.reshape(data_sp['by'],(nx,ny,nz))
    bz = np.reshape(data_sp['bz'],(nx,ny,nz))

    # print(vy[0,0:5,0:5])
    # print(vz[0,0:5,0:5])
    t =  data_sp['t_save'][0]
    # tvel = math.fmod(t, 2.0 * ly / (shear * lx))

    data_sp.close()

    vx_analytical = shear_quantity(vx_0, t)*np.exp(sigma*t)
    vy_analytical = shear_quantity(vy_0, t)*np.exp(sigma*t)
    vz_analytical = shear_quantity(vz_0, t)*np.exp(sigma*t)

    bx_analytical = shear_quantity(bx_0, t)*np.exp(sigma*t)
    by_analytical = shear_quantity(by_0, t)*np.exp(sigma*t)
    bz_analytical = shear_quantity(bz_0, t)*np.exp(sigma*t)



    fig, ax = plt.subplots(3,2, sharex=True, sharey=True)

    im00 = ax[0,0].imshow(vx_analytical[:,0,:].T, origin='lower' ,cmap='RdBu_r',extent=extent, interpolation='none')
    im01 = ax[0,1].imshow(vx[:,0,:].T, origin='lower', cmap='RdBu_r',extent=extent, interpolation='none')

    im10 = ax[1,0].imshow(vy_analytical[:,0,:].T, origin='lower' ,cmap='RdBu_r',extent=extent, interpolation='none')
    im11 = ax[1,1].imshow(vy[:,0,:].T, origin='lower', cmap='RdBu_r',extent=extent, interpolation='none')

    im20 = ax[2,0].imshow(vz_analytical[:,0,:].T, origin='lower' ,cmap='RdBu_r',extent=extent, interpolation='none')
    im21 = ax[2,1].imshow(vz[:,0,:].T, origin='lower', cmap='RdBu_r',extent=extent, interpolation='none')

    ax[0,0].set_title(r'analytical')
    ax[0,1].set_title(r'spooky')

    cbar = fig.colorbar(im00, ax=ax[0,0], orientation='horizontal')
    cbar.set_label(r'$v_x$')

    cbar = fig.colorbar(im01, ax=ax[0,1], orientation='horizontal')
    cbar.set_label(r'$v_x$')

    cbar = fig.colorbar(im10, ax=ax[1,0], orientation='horizontal')
    cbar.set_label(r'$v_y$')

    cbar = fig.colorbar(im11, ax=ax[1,1], orientation='horizontal')
    cbar.set_label(r'$v_y$')

    cbar = fig.colorbar(im20, ax=ax[2,0], orientation='horizontal')
    cbar.set_label(r'$v_z$')

    cbar = fig.colorbar(im21, ax=ax[2,1], orientation='horizontal')
    cbar.set_label(r'$v_z$')
    # ax[0].legend()

    # ax[1].plot(z, by_analytical[0,0,:], ls='', marker='x', markerfacecolor='none', label='analytical')
    # ax[1].plot(z, by[0,0,:], ls='', marker='s', markerfacecolor='none', label='spooky')
    #
    for ii in range(3):
        for jj in range(2):
            ax[ii,jj].set_xlabel(r'$x$')
            ax[ii,jj].set_ylabel(r'$z$')

    # ax[1].legend()

    fig.suptitle(r'MRI eigenmode: analytical vs spooky')

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
