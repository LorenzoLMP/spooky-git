import subprocess
import argparse
import numpy as np
# from numpy.fft import rfft, irfft, rfftfreq
# import matplotlib.pyplot as plt
import h5py
import glob
import sys
import math

# test script expects the executable as argument

nx = 64
ny = 64
nz = 64

lx = 1.0
ly = 1.5
lz = 1.0

shear = 1.5
omega = 1.0
q = shear/omega

reynolds = 3000
reynolds_m = 3000

# Axisymmetric MRI eigenmode with k_x, k_z, Pm=1
# and vertical background magnetic field


B0z = 0.1
kZ     = 2.0*2.0*np.pi/lz
kX     = 1.0*2.0*np.pi/lx
gamma2 = kZ*kZ/(kZ*kZ + kX*kX)
kappa2 = 2*omega*(2*omega - shear)
omegaA2    = B0z*B0z*kZ*kZ
Omega = omega
S = shear
Omega2 = Omega*Omega

# solve for omega_nu2, with omega_nu = -i * sigma_nu
omega_nu2 = omegaA2 + 0.5*kappa2*gamma2*( 1. - np.sqrt(1. + 16.*omegaA2*Omega2/(kappa2*kappa2*gamma2) ) )
sigma_nu = np.sqrt(-omega_nu2)
sigma = sigma_nu - (kX*kX + kZ*kZ)/reynolds
# sigma_nu   = sigma + (kX*kX + kZ*kZ)/reynolds
# we assume sigma_nu and sigma_eta are equal
sigma_eta  = sigma + (kX*kX + kZ*kZ)/reynolds_m


x = -lx/2 + lx * (np.arange(nx)) / nx
y = -ly/2 + ly * (np.arange(ny)) / ny
z = -lz/2 + lz * (np.arange(nz)) / nz

X, Y, Z = np.meshgrid(x,y,z,indexing='ij')

kx_vec = 2.0*np.pi*np.fft.fftfreq(nx, d=lx/nx)
ky_vec = 2.0*np.pi*np.fft.fftfreq(ny, d=ly/ny)
kz_vec = 2.0*np.pi*np.fft.fftfreq(nz, d=lz/nz)

KX, KY, KZ = np.meshgrid(kx_vec, ky_vec, kz_vec, sparse=False, indexing='ij')

a = 1e-4

vx_0 =   a*np.cos(kX*X + kZ*Z)
vy_0 =   vx_0 * (S*omegaA2/(sigma_eta*sigma_eta) - (2*Omega - S))/(sigma_nu + omegaA2/sigma_eta)
vz_0 =   -(kX/kZ)*a*np.cos(kX*X + kZ*Z)

bx_0 = -a*np.sin(kX*X + kZ*Z)*B0z*kZ/sigma_eta
by_0 =  bx_0 * ( - S/(sigma_eta) + (S*omegaA2/(sigma_eta*sigma_eta)   - (2*Omega - S)) /(sigma_nu + omegaA2/sigma_eta) )
bz_0 = B0z * (1.0 + (kX/kZ)*a*np.sin(kX*X + kZ*Z)*kZ/sigma_eta )

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

    data_0.close()

# tol = 1e-15
tol = 1e-7

L1_err = 0.

def shear_quantity(w, tremap):
    # the convention is that this function shears
    # the coords in the same direction of the
    # background flow. If one wants to use it to shear
    # "back" to the last t_remap then t_remap must be <0
    w_sheared = np.real(np.fft.ifft(np.fft.fft(w,axis=1)*np.exp(1j*(KY)*(X)*shear*tremap),axis=1))
    return w_sheared


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


    for i in range(len(sp_data_list)):
    # for i in range(3,4):
        data_sp = h5py.File(sp_savedir+'{:s}{:04d}.h5'.format(sp_savename,i), 'r')
        vx = np.reshape(data_sp['vx'],(nx,ny,nz))
        vy = np.reshape(data_sp['vy'],(nx,ny,nz))
        vz = np.reshape(data_sp['vz'],(nx,ny,nz))

        bx = np.reshape(data_sp['bx'],(nx,ny,nz))
        by = np.reshape(data_sp['by'],(nx,ny,nz))
        bz = np.reshape(data_sp['bz'],(nx,ny,nz))-B0z
        t =  data_sp['t_save'][0]
        data_sp.close()

        vx_analytical = shear_quantity(vx_0, t)*np.exp(sigma*t)
        vy_analytical = shear_quantity(vy_0, t)*np.exp(sigma*t)
        vz_analytical = shear_quantity(vz_0, t)*np.exp(sigma*t)

        bx_analytical = shear_quantity(bx_0, t)*np.exp(sigma*t)
        by_analytical = shear_quantity(by_0, t)*np.exp(sigma*t)
        bz_analytical = shear_quantity(bz_0-B0z, t)*np.exp(sigma*t)

        L1_err = np.sum(np.abs(vx-vx_analytical))
        L1_err += np.sum(np.abs(vy-vy_analytical))
        L1_err += np.sum(np.abs(vz-vz_analytical))

        L1_err += np.sum(np.abs(bx-bx_analytical))
        L1_err += np.sum(np.abs(by-by_analytical))
        L1_err += np.sum(np.abs(bz-bz_analytical))

        L1_err /= (nx*ny*nz)

        print('t = {:10.8f} \t L1 error = {:0.6e}'.format(t,L1_err))

    if (L1_err < tol):
        print('t_final = %10.8f \t L1 error = %.6e ... PASSED'%(t,L1_err))
        flag = 0 # pass
    else:
        print('t_final = %10.8f \t L1 error = %.6e ... NOT PASSED'%(t,L1_err))

    return flag


if __name__ == '__main__':
    result = main()
    assert result == 0 #pass


