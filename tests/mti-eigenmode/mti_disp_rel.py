#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:57:14 2020

@author: lorenzolmp
"""

import os
#import sys
import numpy as np
import matplotlib.pyplot as plt
#import glob
from scipy.optimize import curve_fit
from matplotlib.ticker import FuncFormatter
from pylab import cm
from matplotlib.legend_handler import HandlerLine2D

testdir = './'

#testdir1 = '/home/lorenzolmp/snoopy6/data_temp/test_parasitic_N21e-1_ReyChi1e3_eig_noise/'

# SAVEFIG = False
# #SAVEFIG = True
#
# if(not os.path.isdir(testdir+'Plots/')):
#     os.mkdir(testdir+'Plots/')
    
    
# plt.close('all')    

plt.rc('font', family='serif')
#mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5
colors = cm.get_cmap('tab10', 10)
#chi = 1.0/1e3
#nu  = 1.0/1e6
#eta = 1.0/1e6
#sigmaT = 0.0742327
#k0 = 2.0*np.pi*1.0
#Pe    = 1.0/chi

Pe     = 1e2
ReNu   = 1e5
ReEta  = 1e5
B0     = 1e-4
N2     = 0.1
Pr     = Pe/ReNu  # Pr     = Pe/Renu
q      = Pe/ReEta  # q      = Pe/Reeta
Lambda = B0**2*Pe # Lambda = va^2*Pe

coeff = np.zeros(4)


n_modes = [12]
k2_sim = (2.0*np.pi*np.array(n_modes))**2/Pe

mti_root_sim = np.zeros([k2_sim.shape[0]],dtype='float64')


for i in range(len(k2_sim)):
    k2 = k2_sim[i]
    coeff[0] = 1
    coeff[1] = k2*(1.0+q+Pr)
    coeff[2] = N2+k2*Lambda+(q+Pr)*k2**2+q*Pr*k2**2
    coeff[3] = q*Pr*k2**3 + N2*q*k2 + Lambda*k2**2 - k2
    
    roots_tmp = np.roots(coeff)
    mti_root_sim[i] =  np.max(np.real(roots_tmp[np.imag(roots_tmp)==0]))
   

# fig,ax = plt.subplots()


