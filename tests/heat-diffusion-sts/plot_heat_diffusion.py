import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob

sp_savedir = "../../build/tests/heat-diffusion-sts/data/"
sp_savename = 'snap'

sp_data_list = glob.glob(sp_savedir+'*.h5')

nx = 256
ny = 1
nz = 2

lx = 1.0
ly = 1.0
lz = 1.0

nu_th = 1./100.
sigma = 0.1


x = -0.5 + lx * (np.arange(nx)) / nx
y = -0.5 + ly * (np.arange(ny)) / ny
z = -0.5 + lz * (np.arange(nz)) / nz

X, Y, Z = np.meshgrid(x,y,z,indexing='ij')

def T_analytical_func(X,t):
    fac = (1. + 2 * nu_th * t / sigma**2)
    T =   1./np.sqrt(fac)*np.exp(- X**2/(2*sigma**2*fac))
    return T


for i in range(len(sp_data_list)):
# for i in range(1,2):
# for i in range(3,4):


    data_sp = h5py.File(sp_savedir+'{:s}{:04d}.h5'.format(sp_savename,i), 'r')
    th = np.reshape(data_sp['th'],(nx,ny,nz))

    t =  data_sp['t_save'][0]
    data_sp.close()

    fig, ax = plt.subplots()
    ax.plot(x, T_analytical_func(x,t), color='k', label='theo')
    ax.plot(X[:,0,0],th[:,0,0],label='spooky')

    ax.legend()

plt.show()

# fig, ax = plt.subplots()
#
# for t in np.linspace(0.0,0.1,5):
#     ax.plot(x, T_analytical_func(x,t), color='k', label='t=%.4f'%(t),alpha=1.-t)
#
# ax.legend()
#
# fig.show()
