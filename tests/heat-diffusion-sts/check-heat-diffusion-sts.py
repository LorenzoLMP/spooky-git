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
ny = 1
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
    return T



tol = 1e-7
flag = 1 # fail


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

    # for i in range(len(sp_data_list)):
    i = len(sp_data_list) - 1
    # for i in range(3,4):
    data_sp = h5py.File(sp_savedir+'{:s}{:04d}.h5'.format(sp_savename,i), 'r')
    T = np.reshape(data_sp['th'],(nx,ny,nz))
    t =  data_sp['t_save'][0]

    data_sp.close()

    T_analytical = T_analytical_func(t)

    L1_err = np.sum(np.abs(T-T_analytical))/(nx*ny*nz)

    print('t = {:10.4f} \t L1 error = {:0.6e}'.format(t,L1_err))

    if (L1_err < tol):
        print('t_final = %.4f \t L1 error = %.6e ... PASSED'%(t,L1_err))
        flag = 0 # pass
    else:
        print('t_final = %.4f \t L1 error = %.6e ... NOT PASSED'%(t,L1_err))
        flag = 1 # not pass

    return flag


if __name__ == '__main__':
    result = main()
    assert result == 0 #pass


