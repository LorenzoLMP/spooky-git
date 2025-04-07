import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob

# sp_savedir = "../../build/tests/heat-diffusion-sts/data/"
sp_savedir = "/home/lorenzolmp/Documents/cuda-code/spooky-git/build/tests/heat-diffusion-sts/rkl2/nx_000512/data/"
sp_savename = 'snap'

sp_data_list = glob.glob(sp_savedir+'*.h5')

nx = 512
ny = 2
nz = 2

lx = 4.0
ly = 1.0
lz = 1.0

nu_th = 1./1.
sigma = 0.1


x = -0.5*lx + lx * (np.arange(nx)) / nx
y = -0.5*ly + ly * (np.arange(ny)) / ny
z = -0.5*lz + lz * (np.arange(nz)) / nz

X, Y, Z = np.meshgrid(x,y,z,indexing='ij')

kx = 2.0*np.pi*np.fft.fftfreq(nx, d=lx/nx)
ky = 2.0*np.pi*np.fft.fftfreq(ny, d=ly/ny)
kz = 2.0*np.pi*np.fft.rfftfreq(nz, d=lz/nz)

KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

K2 = KX**2 + KY**2 + KZ**2

T_0 = 1./(np.sqrt(2.*np.pi*sigma**2))*np.exp(-0.5*(X**2)/(sigma**2))

T_0_hat = np.fft.rfftn(T_0)

def T_analytical_func(t):
    T =  np.fft.irfftn(T_0_hat*np.exp(-nu_th*KX**2*t))
    # T = 1./(np.sqrt(2.*np.pi*(sigma**2 + 2 * nu_th*t)))*np.exp(-0.5*(X**2)/(sigma**2 + 2 * nu_th*t))
    return T

for i in range(len(sp_data_list)):
# for i in range(1,2):
# for i in range(3,4):


    data_sp = h5py.File(sp_savedir+'{:s}{:04d}.h5'.format(sp_savename,i), 'r')
    th = np.reshape(data_sp['th'],(nx,ny,nz))

    t =  data_sp['t_save'][0]
    data_sp.close()

    fig, ax = plt.subplots()
    ax.plot(x, T_analytical_func(t)[:,0,0], color='k', label='theo')
    ax.plot(X[:,0,0],th[:,0,0],label='spooky')

    ax.legend()

plt.show()

# fig, ax = plt.subplots()
#
# for t in np.linspace(0.0,0.5,5):
#     ax.plot(x, T_analytical_func(x,t), color='k', label='t=%.4f'%(t),alpha=1.-t)
#
# ax.legend()
#
# fig.show()
