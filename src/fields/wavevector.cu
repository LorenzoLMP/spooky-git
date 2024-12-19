#include "define_types.hpp"
// #include "wavevector.hpp"
#include "cufft_routines.hpp"
#include "spooky.hpp"
#include "fields.hpp"
#include "parameters.hpp"
#include "common.hpp"


Wavevector::~Wavevector() {
    // free(kxt);
    // free(ky);
    // free(kz);
    free(all_kvec);
    free(kvec);
    free(mask);

    free(d_kvec);
}


// void Wavevector::init_Wavevector(Parameters *p_in) {
Wavevector::Wavevector(double Lx, double Ly, double Lz) {
    unsigned int idx;

    // scalar_type Lx, scalar_type Ly, scalar_type Lz
    // lx = p_in->lx; ly = p_in->ly; lz = p_in->lz;
    lx = Lx; ly = Ly; lz = Lz;
    // std::printf("baginning of wave\n");
    // all_kvec contains kx ky kz sequentially
    all_kvec = (scalar_type *) malloc( (size_t) sizeof(scalar_type) * ntotal_complex * 3);
    // kvec is array of arrays such that kvec[0] = kx, etc
    kvec = (scalar_type **) malloc( (size_t) sizeof(scalar_type) * 3);
    // init kvec
    // std::printf("before init kvec\n");
    for (int i = 0 ; i < 3 ; i++) {
        kvec[i]   = all_kvec + i*ntotal_complex;
    }

    d_kvec = (scalar_type **) malloc( (size_t) sizeof(scalar_type *) * 3);
    // kxt = (scalar_type *) malloc( (size_t) sizeof(scalar_type) * ntotal_complex);
    // ky = (scalar_type *) malloc( (size_t) sizeof(scalar_type) * ntotal_complex);
    // kz = (scalar_type *) malloc( (size_t) sizeof(scalar_type) * ntotal_complex);
    // kz = (scalar_type *) malloc( (size_t) sizeof(scalar_type) * ntotal_complex);


    mask = (scalar_type *) malloc( (size_t) sizeof(scalar_type) * ntotal_complex);

    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){
            for (int k = 0; k < nz/2 + 1; k++){
                idx = k + (nz/2+1) * ( j + i * ny);
                kvec[KX][idx] = (2.0 * M_PI) / lx * (fmod( (double) i + ( (double) nx / 2) ,  nx ) - (double) nx / 2 );
                kvec[KY][idx]  = (2.0 * M_PI) / ly * (fmod( (double) j + ( (double) ny / 2) ,  ny ) - (double) ny / 2 );
                kvec[KZ][idx]  = (2.0 * M_PI) / lz * (double) k;
            }
        }
    }

    std::printf("Finished filling wavevector\n");


    kxmax = 2.0 * M_PI/ lx * ( ( (double) nx / 2) - 1);
    kymax = 2.0 * M_PI/ ly * ( ( (double) ny / 2) - 1);
    kzmax = 2.0 * M_PI/ lz * ( ( (double) nz / 2) - 1);

    std::printf("Maximum wavenumbers (without dealiasing): kxmax = %.2e  kymax = %.2e  kzmax = %.2e \n",kxmax,kymax, kzmax);

#ifdef DEALIASING
    for (int i = 0; i < nx; i++){
        for (int j = 0; j < ny; j++){
            for (int k = 0; k < nz/2 + 1; k++){
                idx = k + (nz/2+1) * ( j + i * ny);
                mask[idx] = 1.0;
                if( fabs( kvec[KX][ idx] ) > 2.0/3.0 * kxmax)
                    mask[idx] = 0.0;
                if( fabs( kvec[KY][ idx ] ) > 2.0/3.0 * kymax)
                    mask[idx] = 0.0;
                if( fabs( kvec[KZ][ idx ] ) > 2.0/3.0 * kzmax)
                    mask[idx] = 0.0;
            }
        }
    }
    kxmax = (2.0 / 3.0 ) * kxmax;
    kymax = (2.0 / 3.0 ) * kymax;
    kzmax = (2.0 / 3.0 ) * kzmax;

#endif

    kmax  = pow(kxmax*kxmax+kymax*kymax+kzmax*kzmax,0.5);
}

void Wavevector::print_values() {
    int idx;
    const char* k_comp[3] = {"kx", "ky", "kz"};
    // for (int i = 0; i < 25; i++){
    //     idx =  (nz/2+1)*2 * ( i * ny);
    //     // std::printf("v1[%d]= %f \t v2[%d]= %f \n", idx, farray_r[0][idx], idx, farray_r[1][idx]);
    //     for (int n = 0; n < num_fields; n++){
    //         std::printf("v[%d][%d] = %.7e \t", n, idx, farray_r[n][idx]);
    //     }
    //     std::cout << std::endl;
    // }
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++){
            for (int k = 0; k < 2; k++){
                idx = k + (nz/2+1)*2 * ( j + i * ny);
                // std::printf("v1[%d]= %f \t v2[%d]= %f \n", idx, farray_r[0][idx], idx, farray_r[1][idx]);
                for (int n = 0; n < 3; n++){
                    std::printf("k[%s][%d] = %.3e \t", k_comp[n], idx, kvec[n][idx]);
                }
                std::cout << std::endl;
            }
        }
    }
    // std::printf("kx:\n");
    // int i,j,k, idx;
    // j = 0; k = 0;
    // for (i = 0; i < 10; i++){
    //     idx = k + (nz/2+1) * ( j + i * ny);
    //     std::printf("kx[%d] %f \n", i, kvec[KX][idx]);
    // }
    // i = 0; k = 0;
    // std::printf("ky:\n");
    // for (j = 0; j < 10; j++){
    //     idx = k + (nz/2+1) * ( j + i * ny);
    //     std::printf("ky[%d] %f \n", j, kvec[KY][idx]);
    // }
    // i = 0; j = 0;
    // std::printf("kz:\n");
    // for (k = 0; k < 10; k++){
    //     idx = k + (nz/2+1) * ( j + i * ny);
    //     std::printf("kz[%d] %f \n", k, kvec[KZ][idx]);
    // }
}

void Wavevector::shear_Wavevector( double t, double dt) {
    // write routines for shearing kxt
}

void Wavevector::allocate_and_move_to_gpu() {
    // void *d_vx, *d_vy;

    CUDA_RT_CALL(cudaMalloc(&d_all_kvec, (size_t) sizeof(scalar_type) * ntotal_complex * 3));

    CUDA_RT_CALL(cudaMalloc(&d_mask,  (size_t) sizeof(scalar_type) * ntotal_complex));

    CUDA_RT_CALL(cudaMemcpy(d_all_kvec, all_kvec, (size_t) sizeof(scalar_type) * ntotal_complex * 3, cudaMemcpyHostToDevice));

    CUDA_RT_CALL(cudaMemcpy(d_mask, mask, (size_t) sizeof(scalar_type) * ntotal_complex, cudaMemcpyHostToDevice));

    for (int i = 0 ; i < 3 ; i++) {
        d_kvec[i]  = d_all_kvec + i*ntotal_complex;
    }

}

void Wavevector::sync_with_host() {
    CUDA_RT_CALL(cudaMemcpy(kvec[KX], d_kvec[KX], (size_t) sizeof(scalar_type) * ntotal_complex, cudaMemcpyDeviceToHost));

}

void Wavevector::clean_gpu(){
    CUDA_RT_CALL(cudaFree(d_all_kvec));
    CUDA_RT_CALL(cudaFree(d_mask));
}

