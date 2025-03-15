# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:23:20 2020

@author: lmp61
"""

import os
#import sys
import numpy as np
import matplotlib.pyplot as plt
#import glob
# %matplotlib ipympl
from scipy.optimize import curve_fit
from matplotlib.ticker import FuncFormatter
from pylab import cm
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.lines as mlines

testdir = '../../build/tests/advected-mhd-vortex/'

savedir = testdir

# SAVEFIG = False
SAVEFIG = True

if(not os.path.isdir(savedir+'Plots/')):
    os.mkdir(savedir+'Plots/')


# plt.close('all')


values = np.loadtxt(testdir+"data/"+"timevar.spooky", comments="#", skiprows=0, delimiter=None, unpack=False)

with open(testdir+"data/"+"timevar.spooky") as f:
    # a = dict(i.rstrip().split(None, 1) for i in f)
    content = f.readlines()
    keys = content[1].lstrip('#').rstrip('\n').split('\t')[1:-1]
# print(a)

data = {keys[i]: values[:,i] for i in range(len(keys))}


# test script expects the executable as argument

nx = 512
ny = 512
nz = 2

lx = 1.0
ly = 1.0
lz = 1.0

nu = 1./10000.
eta = 1./10000.

v0 = 0.05
u0 = 0.05

sigma = 200.

x = -lx/2 + lx * (np.arange(nx)) / nx
y = -ly/2 + ly * (np.arange(ny)) / ny
z = -lz/2 + lz * (np.arange(nz)) / nz

X, Y, Z = np.meshgrid(x,y,z,indexing='ij')

def vel_analytical(X,Y,Z,t):
    Xt = X - u0*t
    Yt = Y - v0*t
    R2t = Xt**2 + Yt**2
    vx =   v0 - 1./(2.0 * np.pi) * np.exp( (1.0  - sigma*R2t)/2.0 ) * Yt
    vy =   u0 + 1./(2.0 * np.pi) * np.exp( (1.0  - sigma*R2t)/2.0 ) * Xt
    vz =   np.zeros(X.shape)

    # vx *= np.exp(-2.0*nu*t*(2.0*np.pi)**2)
    # vy *= np.exp(-2.0*nu*t*(2.0*np.pi)**2)
    # vz *= np.exp(-2.0*nu*t*(2.0*np.pi)**2)

    return vx, vy, vz

def mag_analytical(X,Y,Z,t):
    Xt = X - u0*t
    Yt = Y - v0*t
    R2t = Xt**2 + Yt**2
    Bx =   - 1./(2.0 * np.pi) * np.exp( (1.0  - sigma*R2t)/2.0 ) * Yt
    By =     1./(2.0 * np.pi) * np.exp( (1.0  - sigma*R2t)/2.0 ) * Xt
    Bz =   np.zeros(X.shape)

    # Bx *= np.exp(-2.0*eta*t*(2.0*np.pi)**2)
    # By *= np.exp(-2.0*eta*t*(2.0*np.pi)**2)
    # Bz *= np.exp(-2.0*eta*t*(2.0*np.pi)**2)

    return Bx, By, Bz


vx_0, vy_0, vz_0 = vel_analytical(X,Y,Z,0.0)
bx_0, by_0, bz_0 = mag_analytical(X,Y,Z,0.0)


fig, ax = plt.subplots()

ax.plot(data['t'], data['ev'])

plt.show()



freq1Dx = 2.0*np.pi*np.fft.fftfreq(nx, d=lx/nx)
freq1Dy = 2.0*np.pi*np.fft.fftfreq(ny, d=ly/ny)
rfreq1Dz = 2.0*np.pi*np.fft.rfftfreq(nz, d=lz/nz)

rKX, rKY, rKZ = np.meshgrid(freq1Dx, freq1Dy, rfreq1Dz, sparse=False, indexing='ij')

K2 = rKX**2 + rKY**2 + rKZ*2

ik2 = np.where(K2>0, 1/K2, 0.0)


kxmax = 2.0 * np.pi/ lx * ( (nx / 2) - 1)
kymax = 2.0 * np.pi/ ly * ( (ny / 2) - 1)
kzmax = 2.0 * np.pi/ lz * ( (nz / 2) - 1)

mask = np.ones(rKX.shape)
mask[np.abs(rKX)>(2.0/3.0)*kxmax] = 0.0
mask[np.abs(rKY)>(2.0/3.0)*kymax] = 0.0
mask[np.abs(rKZ)>(2.0/3.0)*kzmax] = 0.0

B_0 = np.sqrt(bx_0**2 + by_0**2 + bz_0**2)

bcurl_x = -1j* ( rKY * np.fft.rfftn(bz_0) - rKZ * np.fft.rfftn(by_0))
bcurl_y = -1j* ( rKZ * np.fft.rfftn(bx_0) - rKX * np.fft.rfftn(bz_0))
bcurl_z = -1j* ( rKX * np.fft.rfftn(by_0) - rKY * np.fft.rfftn(bx_0))

hat_hel_x = - ik2 * bcurl_x
hat_hel_y = - ik2 * bcurl_y
hat_hel_z = - ik2 * bcurl_z

hel_x = np.fft.irfftn(hat_hel_x)
hel_y = np.fft.irfftn(hat_hel_y)
hel_z = np.fft.irfftn(hat_hel_z)


helicity = np.sum(bx_0*hel_x)/(nx*ny*nz)
helicity += np.sum(by_0*hel_y)/(nx*ny*nz)
helicity += np.sum(bz_0*hel_z)/(nx*ny*nz)



cross_helicity = np.sum(bx_0*vx_0)/(nx*ny*nz)
cross_helicity += np.sum(by_0*vy_0)/(nx*ny*nz)
cross_helicity += np.sum(bz_0*vz_0)/(nx*ny*nz)


hat_vcurl_x = -1j* ( rKY * np.fft.rfftn(vz_0) - rKZ * np.fft.rfftn(vy_0))
hat_vcurl_y = -1j* ( rKZ * np.fft.rfftn(vx_0) - rKX * np.fft.rfftn(vz_0))
hat_vcurl_z = -1j* ( rKX * np.fft.rfftn(vy_0) - rKY * np.fft.rfftn(vx_0))

vort_x = np.fft.irfftn(hat_vcurl_x)
vort_y = np.fft.irfftn(hat_vcurl_y)
vort_z = np.fft.irfftn(hat_vcurl_z)

enstrophy = 0.5*np.sum(vort_x*vort_x)/(nx*ny*nz)
enstrophy += 0.5*np.sum(vort_y*vort_y)/(nx*ny*nz)
enstrophy += 0.5*np.sum(vort_z*vort_z)/(nx*ny*nz)
