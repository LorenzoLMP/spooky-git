import subprocess
import argparse
import numpy as np
# from numpy.fft import rfft, irfft, rfftfreq
# import matplotlib.pyplot as plt
import h5py
import glob
import sys
# test script expects the executable as argument

nx = 64
ny = 64
nz = 64

lx = 1.0
ly = 1.0
lz = 1.0

nu = 1./200.

x = -0.5 + lx * (np.arange(nx)) / nx
y = -0.5 + ly * (np.arange(ny)) / ny
z = -0.5 + lz * (np.arange(nz)) / nz

X, Y, Z = np.meshgrid(x,y,z,indexing='ij')

# vx_0 =   np.sin(2.0*np.pi*X/lx) * np.cos(2.0*np.pi*Y/ly)
# vy_0 = - np.cos(2.0*np.pi*X/lx) * np.sin(2.0*np.pi*Y/ly)
# vz_0 =   np.zeros(X.shape)

vx_0 =   np.zeros(X.shape)
vy_0 =   np.sin(2.0*np.pi*Y/ly) * np.cos(2.0*np.pi*Z/lz)
vz_0 = - np.cos(2.0*np.pi*Y/ly) * np.sin(2.0*np.pi*Z/lz)

tol = 1e-7



def main():

    flag = 1 # fail

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

    try:
        subprocess.run(
            [args.executable,"--input-dir",args.input_dir,"--output-dir",args.output_dir], timeout=1000, check=True
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
        t =  data_sp['t_save'][0]
        data_sp.close()

        vx_analytical = vx_0 * np.exp(-2.0*nu*t*(2.0*np.pi)**2)
        vy_analytical = vy_0 * np.exp(-2.0*nu*t*(2.0*np.pi)**2)
        vz_analytical = vz_0 * np.exp(-2.0*nu*t*(2.0*np.pi)**2)

        L1_err = np.sum(np.abs(vx-vx_analytical))
        L1_err += np.sum(np.abs(vy-vy_analytical))
        L1_err += np.sum(np.abs(vz-vz_analytical))

        L1_err /= (nx*ny*nz)

        print('t = {:10.4f} \t L1 error = {:0.2e}'.format(t,L1_err))

    if (L1_err < tol):
        print('t_final = %10.4f \t L1 error = %.2e ... PASSED'%(t,L1_err))
        flag = 0 # pass
    else:
        print('t_final = %10.4f \t L1 error = %.2e ... NOT PASSED'%(t,L1_err))

    return flag


if __name__ == '__main__':
    result = main()
    assert result == 0 #pass


