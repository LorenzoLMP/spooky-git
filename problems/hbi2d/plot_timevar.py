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

testdir = '/home/lorenzolmp/Documents/cuda-code/spooky-data/hbi2d/'

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



chi = 1.0/1600.0
N2  = 0.2
nu  = 1.0/12800
eta = 1.0/12800
Pe = 1.0/chi
# alpha = 1e-3
Lx = 1
Ly = 1
Lz = 1
Nx = 512
Ny = 512
Nz = 2
kxmax = 2.0 * np.pi / Lx * ( (Nx / 2) - 1)
kymax = 2.0 * np.pi / Ly * ( (Ny / 2) - 1)
kzmax = 2.0 * np.pi / Lz * ( (Nz / 2) - 1)
kmax=(kxmax*kxmax+kymax*kymax+kzmax*kzmax)**0.5
kmax = kmax * 2.0 / 3.0
# hyp_diff_chi = 1.0/ ( 5e3 * kmax * kmax * kmax * kmax )



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

# fig, ax = plt.subplots()
#
# ax.plot(data['t'], data['ev'])
#
# plt.show()

timevar_vars = data.keys()

for var in timevar_vars:

    fig, ax = plt.subplots()
    ax.plot(data['t'], data[var])

    ax.set_title(f'{var} vs t')
    ax.set_xlabel('t')
    ax.set_ylabel(f'{var}')

plt.show()
