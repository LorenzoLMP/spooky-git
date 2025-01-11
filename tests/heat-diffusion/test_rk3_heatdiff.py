import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
import argparse

nx = 1024
ny = 2
nz = 2

cfl = 1.5

lx = 1.0
ly = 1.0
lz = 1.0

x = -0.5 + lx * (np.arange(nx)) / nx

kxmax = 2.0*np.pi/lx * (nx/2-1)
kymax = 2.0*np.pi/ly * (ny/2-1)
kzmax = 2.0*np.pi/lz * (nz/2-1)

k2max =  kxmax*kxmax + kymax*kymax + kzmax*kzmax

kx = 2*np.pi*rfftfreq(nx, lx/nx)
k2 = kx**2

# data = np.zeros([nx])
ddata = np.zeros([nx])
scratch = np.zeros([nx])

# dt = 0.5
gammaRK = [8.0 / 15.0 , 5.0 / 12.0 , 3.0 / 4.0]
xiRK = [-17.0 / 60.0 , -5.0 / 12.0]

chi0 = 1e-3

def get_chi(T):
    return chi0

# def get_dt(T):
#     gamma_v = np.max(get_chi(T))*k2max
#     dt = cfl/gamma_v
#     return dt

def get_dt():
    gamma_v = chi0*k2max
    dt = cfl/gamma_v
    return dt

def step(T_hat):

    # for when we have spat dep chi
    # T = irfft(T_hat)
    # chi = get_chi(T)
    # dTdx = irfft(1j*kx*T_hat)
    # remove dt here next
    # dT_hat = dt*1j*kx*rfft(chi*dTdx)
    # dT_hat = two_third_dealias_complex_to_complex(dT_hat)

    chi = chi0
    dT_hat = - chi * ( k2 ) * T_hat



    return dT_hat

def RK3(data,dt):

    # RK1:
    ddata = step(data)
    data = data + gammaRK[0] * dt * ddata
    scratch = data + xiRK[0] * dt * ddata

    # RK2
    ddata = step(data)
    data = scratch + gammaRK[1] * dt * ddata
    scratch = data + xiRK[1] * dt * ddata

    # RK3
    ddata = step(data)
    data = scratch + gammaRK[2] * dt * ddata

    return data


a = 0.01
delta = 1.0
output_rate = 5.0
dt_saverate = 5.0
# tend = 10
tend = 10.0
save = True
show = False

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input-dir',
                        default="./",
                        # action='store_const',
                        help='full path to config')
    parser.add_argument('--output-dir',
                        # action='store_const',
                        help='full path to output dir')
    # parser.add_argument('--short',
    #                     default=False,
    #                     action='store_true',
    #                     help='run a shorter test')
    # args = parser.parse_args(['--input-dir'])
    args = parser.parse_args()
    print("test dir is %s"%(args.input_dir))


    savedir = args.output_dir+"/data/"
    savename = 'py_snap'

    t = 0.0
    j = 0
    t_lastsave = 0.0
    numsave = 0

    T = 1 + delta / 2 * (np.tanh((x + 0.375) / a) - np.tanh((x + 0.125) / a)) \
        + delta / 2 * (np.tanh((x - 0.125) / a) - np.tanh((x - 0.375) / a))

    T_hat = rfft(T)

    np.savez(savedir+'{:s}.{:04d}.npz'.format(savename,numsave), x=x, T=T, t=t, dt=tend-t)

    if show:
        plt.figure(1)
        plt.clf()
        fig, axes = plt.subplots(num=1, nrows=1,figsize=(10,10))
        im1, = axes.plot(x, T, '-')
        # im2, = axes[1].plot(x, get_chi(T), '-')
        # im3, = axes[2].plot(x, beta, '-')
        # im4, = axes[3].loglog(kx[1:], np.abs(T_hat[1:])**2, '-')
        # axes[3].loglog(kx[1:], np.max(np.abs(T_hat[1:])**2)*np.exp(-mudt*(kx[1:]/kg)**(2*alpha)))
        # plt.show()
        plt.show(block = False)
        axes.set_ylabel(r'$T$')
        # axes[1].set_ylabel(r'$\chi$')
        # axes[2].set_ylabel(r'$\beta$')
        # axes[2].set_xlabel(r'$x$')

    while (t < tend):
        dt = get_dt()
        if (t+dt > tend): dt = tend - t
        # Update time
        t += dt


        # euler timestepping
        # dT_hat, chi = step(T_hat,dt)
        # T_hat += dT_hat
        # runge-kutta timestepping
        T_hat = RK3(T_hat,dt)

        if save:
            if (t - t_lastsave >= dt_saverate):
                print("saving at time t = %.6e"%(t))
                T[:] = irfft(T_hat)
                np.savez(savedir+'{:s}.{:04d}.npz'.format(savename,numsave+1), x=x, T=T, t=t, dt=tend-t)
                t_lastsave = t_lastsave + dt_saverate;
                numsave += 1
        if show:
            if (t >= output_rate*j-dt):
                T[:] = irfft(T_hat)
                j += 1
                im1.set_ydata(T)
                # chi = get_chi(T)
                # im2.set_ydata(chi)
                # im4.set_ydata(np.abs(T_hat[1:])**2)

                axes.set_title("time {:1.2f}".format(t)+" dt {:1.5f}".format(dt))
                plt.pause(1e-3)
    # plt.savefig("no_reduction_Nx_{}.pdf".format(Nx))

    # # final dt to stop precisely at t = 100
    # dT_hat, chi = step(T_hat,tend-t)
    # T_hat += dT_hat
    print("dt=",dt)
    T[:] = irfft(T_hat)
    # j += 1
    # if save:
    #     np.savez(savedir+'{:s}.{:04d}.npz'.format(savename,numsave+1), x=x, T=T, t=tend, dt=tend-t)
    if show:
        im1.set_ydata(T)
        # chi = get_chi(T)
        # im2.set_ydata(chi)
        # im4.set_ydata(np.abs(T_hat[1:])**2)

        axes.set_title("{:1.4f}".format(tend))
        plt.pause(1e-3)

    return

if __name__ == '__main__':
    main()

