import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
import h5py
import glob
import subprocess
import argparse
import sys

py_savename = 'py_snap'
sp_savename = 'snap'



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

tol = 1e-15

a = 0.01
delta = 1.0

th_0 = 1 + delta / 2 * (np.tanh((X + 0.375) / a) - np.tanh((X + 0.125) / a)) \
        + delta / 2 * (np.tanh((X - 0.125) / a) - np.tanh((X - 0.375) / a))


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

    dset = data_0.create_dataset("th", (nx*ny*nz,), dtype='f8')
    dset[:] = th_0.flatten()

    data_0.close()


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

    args = parser.parse_args()
    print("test dir is %s"%(args.input_dir))

    py_savedir = args.output_dir+"/data/"
    sp_savedir = args.output_dir+"/data/"

    # create ICs
    createICs(args.output_dir)

    try:
        subprocess.run(
            ["python", args.input_dir+"/"+"test_rk3_heatdiff.py","--input-dir",args.input_dir,"--output-dir",args.output_dir], timeout=1000, check=True )
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



    py_data_list = glob.glob(py_savedir+'*.npz')
    sp_data_list = glob.glob(sp_savedir+'*.h5')

    if (len(py_data_list) != len(sp_data_list)):
        print('Number of datafiles not the same!')

    for i in range(len(py_data_list)):
    # for i in range(3,4):
        data_py = np.load(py_savedir+'{:s}.{:04d}.npz'.format(py_savename,i))
        x_py = data_py['x']
        T_py = data_py['T']
        data_py.close()

        data_sp = h5py.File(sp_savedir+'{:s}{:04d}.h5'.format(sp_savename,i), 'r')
        t =  data_sp['t_save'][0]
        T_sp = np.reshape(data_sp['th'],(nx,ny,nz))
        data_sp.close()

        result = 1
        for k in range(nx):
            result *= np.all(T_sp[k,:,:].ravel() == T_sp[k,0,0])
        if not result:
            print('problem: some values in y-z direction are not same')
            return 1

        L1_err = np.sum(np.abs(T_py-T_sp[:,0,0]))
        L1_err /= (nx*ny*nz)

        if (L1_err < tol):
            print('i = %10.4f \t L1 error = %.2e ... PASSED'%(t,L1_err))
        else:
            print('i = %10.4f \t L1 error = %.2e ... NOT PASSED'%(t,L1_err))

    if (L1_err < tol):
        print('t_fin = %10.4f \t L1 error = %.2e ... PASSED'%(t,L1_err))
        flag = 0 #success

    return flag


if __name__ == '__main__':
    result = main()
    assert result == 0 #pass
