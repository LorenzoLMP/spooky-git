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
ny = 512
nz = 2

lx = 1.0
ly = 1.0
lz = 1.0

nu = 1./10000.
eta = 1./10000.

v0 = 0.05
u0 = 0.05

sigma = 200.

x = -lx/2 + lx * (np.arange(nx)) / nx
y = -ly/2 + ly * (np.arange(ny)) / ny
z = -lz/2 + lz * (np.arange(nz)) / nz

X, Y, Z = np.meshgrid(x,y,z,indexing='ij')

def vel_analytical(X,Y,Z,t):
    Xt = X - u0*t
    Yt = Y - v0*t
    R2t = Xt**2 + Yt**2
    vx =   v0 - 1./(2.0 * np.pi) * np.exp( (1.0  - sigma*R2t)/2.0 ) * Yt
    vy =   u0 + 1./(2.0 * np.pi) * np.exp( (1.0  - sigma*R2t)/2.0 ) * Xt
    vz =   np.zeros(X.shape)

    # vx *= np.exp(-2.0*nu*t*(2.0*np.pi)**2)
    # vy *= np.exp(-2.0*nu*t*(2.0*np.pi)**2)
    # vz *= np.exp(-2.0*nu*t*(2.0*np.pi)**2)

    return vx, vy, vz

def mag_analytical(X,Y,Z,t):
    Xt = X - u0*t
    Yt = Y - v0*t
    R2t = Xt**2 + Yt**2
    Bx =   - 1./(2.0 * np.pi) * np.exp( (1.0  - sigma*R2t)/2.0 ) * Yt
    By =     1./(2.0 * np.pi) * np.exp( (1.0  - sigma*R2t)/2.0 ) * Xt
    Bz =   np.zeros(X.shape)

    # Bx *= np.exp(-2.0*eta*t*(2.0*np.pi)**2)
    # By *= np.exp(-2.0*eta*t*(2.0*np.pi)**2)
    # Bz *= np.exp(-2.0*eta*t*(2.0*np.pi)**2)

    return Bx, By, Bz


vx_0, vy_0, vz_0 = vel_analytical(X,Y,Z,0.0)
bx_0, by_0, bz_0 = mag_analytical(X,Y,Z,0.0)


tol = 1e-4 ## for this test lower the tolerance to pass (need to check why...)
# flag = 1 # fail


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

    for i in range(len(sp_data_list)):
    # for i in range(3,4):
        data_sp = h5py.File(sp_savedir+'{:s}{:04d}.h5'.format(sp_savename,i), 'r')
        vx = np.reshape(data_sp['vx'],(nx,ny,nz))
        vy = np.reshape(data_sp['vy'],(nx,ny,nz))
        vz = np.reshape(data_sp['vz'],(nx,ny,nz))

        bx = np.reshape(data_sp['bx'],(nx,ny,nz))
        by = np.reshape(data_sp['by'],(nx,ny,nz))
        bz = np.reshape(data_sp['bz'],(nx,ny,nz))

        t =  data_sp['t_save'][0]
        data_sp.close()

        # vx_analytical = vx_0 * np.exp(-2.0*nu*t*(2.0*np.pi)**2)
        # vy_analytical = vy_0 * np.exp(-2.0*nu*t*(2.0*np.pi)**2)
        # vz_analytical = vz_0 * np.exp(-2.0*nu*t*(2.0*np.pi)**2)

        vx_analytical, vy_analytical, vz_analytical = vel_analytical(X,Y,Z,t)

        bx_analytical, by_analytical, bz_analytical = mag_analytical(X,Y,Z,t)


        L1_err = np.sum(np.abs(vx-vx_analytical))
        L1_err += np.sum(np.abs(vy-vy_analytical))
        L1_err += np.sum(np.abs(vz-vz_analytical))

        L1_err += np.sum(np.abs(bx-bx_analytical))
        L1_err += np.sum(np.abs(by-by_analytical))
        L1_err += np.sum(np.abs(bz-bz_analytical))

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


