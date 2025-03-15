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


# values = np.loadtxt(testdir+"data/"+"timevar.spooky", comments="#", skiprows=0, delimiter=None, unpack=False)
data = {}

with open(testdir+"data/"+"spectrum.spooky") as f:

    for i, line in enumerate(f):
        if (i==0 and line[0] == '#'):
            print(line)
        if (i==1 and line[0] == '#'):
            keys = line.lstrip('##').rstrip('\n').split('\t')[1:-1]
            print(keys)
            for key in keys:
                # data[key] = {'t': [], 'value': []}
                data[key] = {'value': []}
        if (i==2 and line[0:10] == 'wavevector'):
            kvec = np.array(line.rstrip('\n').split('\t')[1:-1], dtype='float64')
            data['kvec'] = kvec

with open(testdir+"data/"+"spectrum.spooky") as f:

    for i, line in enumerate(f):
        if (line[0] == 't'):
            line_split = line.rstrip('\n').split('\t')
            time = float(line_split[1])
            qty = line_split[2]
            qty_vec = np.array(line_split[3:-1], dtype='float64')
            # data[qty]['t'].append(time)
            data[qty]['value'].append(qty_vec)

for key in keys:
    data[key]['value'] = np.array(data[key]['value']).T

fig, ax = plt.subplots()

ax.plot(data['kvec'], data['Kz']['value'])
ax.set_xscale('log')
ax.set_yscale('log')
fig.show()

# print(a)
#
# data = {keys[i]: values[:,i] for i in range(len(keys))}
# content[3].rstrip('\n').split('\t')
#
# for i in range(len(content)):
#     line_tmp = content[i].rstrip('\n').split('\t')
#     if (line_tmp[0] == 't'):
#         timestamp = float(line_tmp[1])
#         qty = line_tmp[2]

# dic[l1[2]] = {'t': float(l1[1]), 'value': np.array(l1[3:-1],dtype='float64')}

# test script expects the executable as argument

nx = 512
ny = 4
nz = 4

lx = 1.0
ly = 1.0
lz = 1.0



freq1Dx = 2.0*np.pi*np.fft.fftfreq(nx, d=lx/nx)
freq1Dy = 2.0*np.pi*np.fft.fftfreq(ny, d=ly/ny)
rfreq1Dz = 2.0*np.pi*np.fft.rfftfreq(nz, d=lz/nz)

rKX, rKY, rKZ = np.meshgrid(freq1Dx, freq1Dy, rfreq1Dz, sparse=False, indexing='ij')

kxmax = 2.0 * np.pi/ lx * ( (nx / 2) - 1)
kymax = 2.0 * np.pi/ ly * ( (ny / 2) - 1)
kzmax = 2.0 * np.pi/ lz * ( (nz / 2) - 1)
