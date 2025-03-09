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

testdir = '../../build/tests/mti-eigenmode/'

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
a = 1e-5

x = -lx/2 + lx * (np.arange(nx)) / nx
y = -ly/2 + ly * (np.arange(ny)) / ny
z = -lz/2 + lz * (np.arange(nz)) / nz

X, Y, Z = np.meshgrid(x,y,z,indexing='ij')

vx_0 = np.zeros(X.shape)
vy_0 = np.zeros(X.shape)
vz_0 = -a*np.sin(kparallel*X)

bx_0 = B0*np.ones(X.shape)
by_0 = np.zeros(X.shape)
bz_0 = -a*np.cos(kparallel*X)*B0*kparallel/(sigma+kparallel*kparallel/Reeta)

th_0 = 1.0/(sigma + kparallel*kparallel/Pe)*(N2 - kparallel*kparallel/Pe/(sigma+kparallel*kparallel/Reeta) ) * vz_0


# chi = 1.0/100.0
# N2  = 0.1
# nu  = 1.0/100000
# eta = 1.0/100000
# Pe = 1.0/chi
# Reeta = 1./eta
# sigma  = 0.9341413811120219
# kparallel  = (2.0*np.pi/lx)*12.0
# B0 = 1e-4


theta_0 = 1.0/(sigma + kparallel*kparallel/Pe)*(N2 - kparallel*kparallel/Pe/(sigma+kparallel*kparallel/Reeta) )

deltaBz_0 = B0*kparallel/(sigma+kparallel*kparallel/Reeta)

# t            = data[:,0]
# ekin         = data[:,1]
# emag         = data[:,2]
# epot         = data[:,3]/N2
# # epot         = data[:,3]/1.0
# ekin_vert    = data[:,6]
# ekin_horiz   = data[:,4]+data[:,5]
# emag_vert    = data[:,9]
# emag_horiz   = data[:,7]+data[:,8]
# thvx         = data[:,10]
# # bzavg        = data[:,13]
# flux_input   = data[:,13]*(chi/N2)
# diss_chi     = data[:,14]*(chi/N2)
# # flux_input   = data[:,13]*(chi/1.0)
# # diss_chi     = data[:,14]*(chi/1.0)
# diss_nu      = 2.0*data[:,15]*(nu)
# diss_eta     = 2.0*data[:,16]*(eta)
# # bz2avg       = data[:,14]
# # diss_hyp_chi = -data[:,33]*hyp_diff_chi/N2
# # chiavg       = data[:,31]
# # chiavg       = np.ones(t.shape)
# # isolength    = data[:,33]
#
# # hv           = data[:,9]
# # hc           = data[:,10]
# # hm           = data[:,34]
# flux_rel = (-flux_input-diss_nu-diss_chi-diss_eta)*Pe*N2

fig, ax = plt.subplots()

ax.plot(data['t'], data['ev'])

plt.show()



freq1Dx = 2.0*np.pi*np.fft.fftfreq(nx, d=lx/nx)
freq1Dy = 2.0*np.pi*np.fft.fftfreq(ny, d=ly/ny)
rfreq1Dz = 2.0*np.pi*np.fft.rfftfreq(nz, d=lz/nz)

rKX, rKY, rKZ = np.meshgrid(freq1Dx, freq1Dy, rfreq1Dz, sparse=False, indexing='ij')

kxmax = 2.0 * np.pi/ lx * ( (nx / 2) - 1)
kymax = 2.0 * np.pi/ ly * ( (ny / 2) - 1)
kzmax = 2.0 * np.pi/ lz * ( (nz / 2) - 1)

mask = np.ones(rKX.shape)
mask[np.abs(rKX)>(2.0/3.0)*kxmax] = 0.0
mask[np.abs(rKY)>(2.0/3.0)*kymax] = 0.0
mask[np.abs(rKZ)>(2.0/3.0)*kzmax] = 0.0

B_0 = np.sqrt(bx_0**2 + by_0**2 + bz_0**2)

bzunitbx_0_hat = np.fft.rfftn((bx_0/B_0)*(bz_0/B_0))
bzunitby_0_hat = np.fft.rfftn((by_0/B_0)*(bz_0/B_0))
bzunitbz_0_hat = np.fft.rfftn((bz_0/B_0)*(bz_0/B_0))

div_bzunitb_hat = bzunitbx_0_hat * 1j*rKX + bzunitby_0_hat * 1j*rKY + bzunitbz_0_hat * 1j*rKZ



div_bzunitb = np.fft.irfftn(div_bzunitb_hat*mask)

inj_aniso = np.sum(th_0*div_bzunitb)/(nx*ny*nz)



