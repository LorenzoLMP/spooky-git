import subprocess
import argparse
import numpy as np
# from numpy.fft import rfft, irfft, rfftfreq
# import matplotlib.pyplot as plt
import h5py
import glob
import sys

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


x = -0.5 + lx * (np.arange(nx)) / nx
y = -0.5 + ly * (np.arange(ny)) / ny
z = -0.5 + lz * (np.arange(nz)) / nz

X, Y, Z = np.meshgrid(x,y,z,indexing='ij')

vx_0 = np.zeros(X.shape)
vy_0 = np.zeros(X.shape)
vz_0 = -0.00001*np.sin(kparallel*X)

bx_0 = B0*np.ones(X.shape)
by_0 = np.zeros(X.shape)
bz_0 = -0.00001*np.cos(kparallel*X)*B0*kparallel/(sigma+kparallel*kparallel/Reeta)

th_0 = 1.0/(sigma + kparallel*kparallel/Pe)*(N2 - kparallel*kparallel/Pe/(sigma+kparallel*kparallel/Reeta) ) * vz_0


tol = 1e-7
# flag = 1 # fail

def createICs(output_dir):

    print("savedir is %s"%(output_dir))
    sp_savename = 'snap'

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

    dset = data_0.create_dataset("vx", (nx*ny*nz,), dtype='f8')
    dset[:] = vx_0.flatten()

    dset = data_0.create_dataset("vy", (nx*ny*nz,), dtype='f8')
    dset[:] = vy_0.flatten()

    dset = data_0.create_dataset("vz", (nx*ny*nz,), dtype='f8')
    dset[:] = vz_0.flatten()

    dset = data_0.create_dataset("bx", (nx*ny*nz,), dtype='f8')
    dset[:] = bx_0.flatten()

    dset = data_0.create_dataset("by", (nx*ny*nz,), dtype='f8')
    dset[:] = by_0.flatten()

    dset = data_0.create_dataset("bz", (nx*ny*nz,), dtype='f8')
    dset[:] = bz_0.flatten()

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

    # create ICs
    createICs(args.output_dir)

    try:
        subprocess.run(
            [args.executable,"--input-dir",args.input_dir,"--output-dir",args.output_dir,"-r","0"], timeout=1000, check=True
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


    sp_savedir = args.output_dir+"/data/"
    sp_savename = 'snap'

    sp_data_list = glob.glob(sp_savedir+'*.h5')

    # if (len(py_data_list) != len(sp_data_list)):
    #     print('Number of datafiles not the same!')

    for i in range(len(sp_data_list)):
    # for i in range(3,4):
        data_sp = h5py.File(sp_savedir+'{:s}{:04d}.h5'.format(sp_savename,i), 'r')
        vx = np.reshape(data_sp['vx'],(nx,ny,nz))
        vy = np.reshape(data_sp['vy'],(nx,ny,nz))
        vz = np.reshape(data_sp['vz'],(nx,ny,nz))

        bx = np.reshape(data_sp['bx'],(nx,ny,nz))
        by = np.reshape(data_sp['by'],(nx,ny,nz))
        bz = np.reshape(data_sp['bz'],(nx,ny,nz))

        th = np.reshape(data_sp['th'],(nx,ny,nz))

        t =  data_sp['t_save'][0]
        data_sp.close()

        # vx_analytical = vx_0 * np.exp(-2.0*nu*t*(2.0*np.pi)**2)
        # vy_analytical = vy_0 * np.exp(-2.0*nu*t*(2.0*np.pi)**2)
        # vz_analytical = vz_0 * np.exp(-2.0*nu*t*(2.0*np.pi)**2)

        vx_analytical = vx_0
        vy_analytical = vy_0
        vz_analytical = vz_0 * np.exp(sigma*t)

        bx_analytical = bx_0
        by_analytical = by_0
        bz_analytical = bz_0 * np.exp(sigma*t)

        th_analytical = th_0 * np.exp(sigma*t)

        L1_err = np.sum(np.abs(vx-vx_analytical))
        L1_err += np.sum(np.abs(vy-vy_analytical))
        L1_err += np.sum(np.abs(vz-vz_analytical))

        L1_err += np.sum(np.abs(bx-bx_analytical))
        L1_err += np.sum(np.abs(by-by_analytical))
        L1_err += np.sum(np.abs(bz-bz_analytical))

        L1_err += np.sum(np.abs(th-th_analytical))

        L1_err /= (nx*ny*nz)

        print('t = {:10.4f} \t L1 error = {:0.2e}'.format(t,L1_err))

    if (L1_err < tol):
        print('t_final = %.4f \t L1 error = %.2e ... PASSED'%(t,L1_err))
        flag = 0 # pass
    else:
        print('t_final = %.4f \t L1 error = %.2e ... NOT PASSED'%(t,L1_err))
        flag = 1 # not pass

    return flag


if __name__ == '__main__':
    result = main()
    assert result == 0 #pass


