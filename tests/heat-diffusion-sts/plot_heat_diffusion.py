import numpy as np
import matplotlib.pyplot as plt


nx = 256
ny = 1
nz = 2

lx = 1.0
ly = 1.0
lz = 1.0

nu_th = 1.
sigma = 0.1


x = -0.5 + lx * (np.arange(nx)) / nx

def T_analytical_func(X,t):
    fac = (1. + 2 * nu_th * t / sigma**2)
    T =   1./np.sqrt(fac)*np.exp(- X**2/(2*sigma**2*fac))
    return T


fig, ax = plt.subplots()

for t in np.linspace(0.0,0.1,5):
    ax.plot(x, T_analytical_func(x,t), color='k', label='t=%.4f'%(t),alpha=1.-t)

ax.legend()

fig.show()
