import subprocess
import argparse
import numpy as np
# from numpy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
import h5py
import glob
import sys
import os
# test script expects the executable as argument


lx = 4.0
ly = 1.0
lz = 1.0

nu_th = 1./1.
sigma = 0.1


def T_analytical_func(T_0_hat, KX, X, t):
    T =  np.fft.irfftn(T_0_hat*np.exp(-nu_th*KX**2*t))
    # T = 1./(np.sqrt(2.*np.pi*(sigma**2 + 2 * nu_th*t)))*np.exp(-0.5*(X**2)/(sigma**2 + 2 * nu_th*t))
    return T


tol = 1e-7
flag = 1 # fail

sts_algorithms = ["sts","rkl1","rkl2"]
# sts_algorithms = ["rkl2"]

def createICs(output_dir, **kwargs):

    print("savedir is %s"%(output_dir))
    sp_savename = 'snap'

    if 'nx' in kwargs.keys():
        nx = kwargs['nx']
    if 'ny' in kwargs.keys():
        ny = kwargs['ny']
    if 'nz' in kwargs.keys():
        nz = kwargs['nz']
    if 'T_0' in kwargs.keys():
        th_0 = kwargs['T_0']

    # with h5py.File(sp_savedir+'{:s}{:04d}.h5'.format(sp_savename,0), 'w') as data_0:
    data_0 = h5py.File(output_dir+'/data/'+'{:s}{:04d}.h5'.format(sp_savename,0), 'w')

    dset = data_0.create_dataset("step", (1,), dtype='i4')
    dset[0] = 0

    dset = data_0.create_dataset("t_end", (1,), dtype='f8')
    dset[0] = 10.0

    dset = data_0.create_dataset("t_lastsnap", (1,), dtype='f8')
    dset[0] = 0.0

    dset = data_0.create_dataset("t_lastvar", (1,), dtype='f8')
    dset[0] = 0.0

    dset = data_0.create_dataset("t_save", (1,), dtype='f8')
    dset[0] = 0.0

    dset = data_0.create_dataset("t_start", (1,), dtype='f8')
    dset[0] = 0.0

    dset = data_0.create_dataset("th", (nx*ny*nz,), dtype='f8')
    dset[:] = th_0.flatten()

    data_0.close()

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--executable',
                        help='full path to executable')

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

    f = open(args.output_dir+"/sts_bench_results.txt", "w")
    f.write("## This file contains the results of the bench test for sts \n")
    f.write("## The table contains the L1 norm of last snapshot (t=0.5) wrt the analytic solution \n")
    f.write("## nx \t sts \t rkl1 \t rkl2 \n")
    f.close()

    for m in range(5, 12):

        nx = 2**m
        ny = 2
        nz = 2

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

        L1_norms = []

        for sts_algo in sts_algorithms:

            print("%s: nx = %06d"%(sts_algo,nx))

            subtest_output_dir = args.output_dir+'/%s/nx_%06d'%(sts_algo,nx)
            if not os.path.exists(subtest_output_dir):
                os.makedirs(subtest_output_dir+'/data',exist_ok=True)
                # os.mkdir(subtest_output_dir+'/data')


            # create ICs
            createICs(subtest_output_dir, nx=nx, ny=ny, nz=nz, T_0=T_0)

            try:
                # subprocess.run(
                #     [args.executable,"--input-dir",args.input_dir+'/%s'%(sts_algo), "--output-dir", subtest_output_dir, "--ngrid", "%d"%(nx), "%d"%(ny), "%d"%(nz), "--stats", "%d"%(2**m+1)], timeout=1000, check=True
                # )
                subprocess.run(
                    [args.executable,"--input-dir",args.input_dir+'/%s'%(sts_algo), "--output-dir", subtest_output_dir, "--ngrid", "%d"%(nx), "%d"%(ny), "%d"%(nz),"-r","0"], timeout=1000, check=True
                )
            except FileNotFoundError as exc:
                print(f"Process failed because the executable could not be found.\n{exc}")
                sys.exit(1)
            except subprocess.CalledProcessError as exc:
                print(
                    f"Process failed because did not return a successful return code. "
                    f"Returned {exc.returncode}\n{exc}"
                )
                sys.exit(1)
            except subprocess.TimeoutExpired as exc:
                print(f"Process timed out.\n{exc}")
                sys.exit(1)


            sp_savedir = subtest_output_dir+"/data/"
            sp_savename = 'snap'

            sp_data_list = glob.glob(sp_savedir+'*.h5')


            i = len(sp_data_list) - 1
            # for i in range(3,4):
            data_sp = h5py.File(sp_savedir+'{:s}{:04d}.h5'.format(sp_savename,i), 'r')
            T = np.reshape(data_sp['th'],(nx,ny,nz))
            t =  data_sp['t_save'][0]

            data_sp.close()

            T_analytical = T_analytical_func(T_0_hat, KX, X, t)

            L1_err = np.sum(np.abs(T-T_analytical))/(nx*ny*nz)

            print('t = {:10.4f} \t L1 error = {:0.6e}'.format(t,L1_err))

            L1_norms.append(L1_err)

        f = open(args.output_dir+"/sts_bench_results.txt", "a")
        f.write("%d \t %.6e \t %.6e \t %.6e \n"%(2**m, L1_norms[0], L1_norms[1], L1_norms[2]))
        f.close()


    return 0

def plot():

    parser = argparse.ArgumentParser()

    parser.add_argument('--executable',
                        help='full path to executable')

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

    bench_results = np.loadtxt(args.output_dir+"/sts_bench_results.txt", comments="#", delimiter=None, unpack=False)
    nx_array = bench_results[:,0]
    L1_sts = bench_results[:,1]
    L1_rkl1 = bench_results[:,2]
    L1_rkl2 = bench_results[:,3]

    xx = np.logspace(np.log10(nx_array[0]), np.log10(nx_array[-1]), 100)

    fig, ax = plt.subplots()

    ax.plot(nx_array, L1_sts, 's', color='r', label='sts')
    ax.plot(nx_array, L1_rkl1, 'x', color='g', label='rkl1')
    ax.plot(nx_array, L1_rkl2, 'o', color='b', label='rkl2')

    ax.plot(xx, L1_sts[0]*(xx/xx[0])**(-1), '--', label=r'$N^{-1}$')
    ax.plot(xx, L1_rkl2[0]*(xx/xx[0])**(-2), '-', label=r'$N^{-2}$')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r'$N_x$')
    ax.set_ylabel(r'$L_1$')

    ax.legend()

    fig.suptitle('Scaling of L1 norms for supertimestepping')

    fig.savefig(args.output_dir+'/sts_benchmark.pdf',dpi=400,transparent=True,bbox_inches='tight')

    plt.close()


    return 0


if __name__ == '__main__':

    result = 1
    result = main()
    plot()



    assert result == 0 #pass


