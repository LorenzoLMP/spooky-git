import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
import h5py
import glob

sp_savedir = "../../build/tests/heat-diffusion/data/"
py_savedir = sp_savedir
sp_savename = 'snap'
py_savename = 'py_snap'

sp_data_list = glob.glob(sp_savedir+'*.h5')

# if (len(py_data_list) != len(sp_data_list)):
#     print('Number of datafiles not the same!')

nx = 1024
ny = 2
nz = 2

lx = 1.0
ly = 1.0
lz = 1.0


x = -0.5 + lx * (np.arange(nx)) / nx
y = -0.5 + ly * (np.arange(ny)) / ny
z = -0.5 + lz * (np.arange(nz)) / nz

X, Y, Z = np.meshgrid(x,y,z,indexing='ij')

# vx_0 =   np.sin(2.0*np.pi*X/lx) * np.cos(2.0*np.pi*Y/ly)
# vy_0 = - np.cos(2.0*np.pi*X/lx) * np.sin(2.0*np.pi*Y/ly)
# vz_0 =   np.zeros(X.shape)
a = 0.01
delta = 1.0


th_0 =   1 + delta / 2 * (np.tanh((X + 0.375) / a) - np.tanh((X + 0.125) / a)) \
        + delta / 2 * (np.tanh((X - 0.125) / a) - np.tanh((X - 0.375) / a))



# e.g. t = 5
# vx_0[0:10,0,0]*np.exp(-2.0*nu*t*(2.0*np.pi)**2)
# vy_0[0:10,10,0]*np.exp(-2.0*nu*5*(2.0*np.pi)**2)

for i in range(len(sp_data_list)):
# for i in range(1,2):
# for i in range(3,4):

    data_py = np.load(py_savedir+'{:s}.{:04d}.npz'.format(py_savename,i))
    x_py = data_py['x']
    T_py = data_py['T']
    data_py.close()

    data_sp = h5py.File(sp_savedir+'{:s}{:04d}.h5'.format(sp_savename,i), 'r')
    th = np.reshape(data_sp['th'],(nx,ny,nz))

    t =  data_sp['t_save'][0]
    data_sp.close()

    fig, ax = plt.subplots()
    ax.plot(x_py,T_py,label='theo')
    ax.plot(X[:,0,0],th[:,0,0],label='spooky')

    ax.legend()

plt.show()




