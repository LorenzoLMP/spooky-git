#include "define_types.hpp"
// #include "wavevector.hpp"
#include "parameters.hpp"





class Wavevector {
// private:
//     scalar_type *r_data;
//     data_type *c_data;
public:
    scalar_type *all_kvec;
    scalar_type **kvec;
    // scalar_type *kxt, *ky, *kz;
    scalar_type *mask;
    scalar_type kxmax, kymax, kzmax, kmax;
    scalar_type *d_all_kvec;
    scalar_type **d_kvec;
    scalar_type *d_mask;
    scalar_type lx, ly, lz;
    // scalar_type *d_kxt, *d_ky, *d_kz, *d_mask;
    // Wavevector() : lx(0.0), ly(0.0), lz(0.0) {};
    // Wavevector();
    Wavevector(scalar_type Lx, scalar_type Ly, scalar_type Lz);
    // Wavevector(scalar_type lx, scalar_type ly, scalar_type lz) : lx{lx}, ly{ly}, lz{lz} {};
    // Wavevector() = default;
    void init_Wavevector();
    void print_values();
    void allocate_and_move_to_gpu();
    void shear_Wavevector( double t, double dt);
    void sync_with_host();
    void clean_gpu();
    ~Wavevector();
};


class Fields {
// private:
//     scalar_type *r_data;
//     data_type *c_data;
public:
    int num_fields;
    int num_tmp_array;
    double current_dt;
    double current_time;
    unsigned int current_step;
    data_type *all_fields, *all_dfields;
    data_type **farray, **dfarray;
    scalar_type **farray_r, **dfarray_r;
    // data_type *vx;
    // data_type *vy;
    data_type *d_all_fields, *d_all_dfields, *d_all_scrtimestep, *d_all_tmparray;
    data_type **d_farray, **d_dfarray, **d_tmparray;
    scalar_type **d_farray_r, **d_dfarray_r, **d_tmparray_r;
    // scalar_type *vx_r, *vy_r;
    Wavevector wavevector;
    // Parameters param;
    Fields( int num, Parameters *param );
    void init_Fields( Parameters *param );
    void init_SpatialStructure(Parameters *param);
    void print_host_values();
    void print_device_values();
    void allocate_and_move_to_gpu();
    void compare_with_original();
    // void do_operations(Wavevector *wavevector);
    // void do_operations();
    // void do_multiplications();
    // double advance_timestep( double t, double t_end, int* p_step);
    void compute_dfield( int stage_step, Parameters *param);
    void compute_dt( Parameters *param );
    void copy_back_to_host();
    void write_data_file( int num_snap, Parameters *param);
    void clean_gpu();
    void RungeKutta3( double t, double t_end, Parameters *param );
    void ComputeDivergence(Parameters *param);
    void CleanFieldDivergence( );
    ~Fields();
};


/*
class Field {
private:
    scalar_type *r_data;
    data_type *c_data;
public:
    Field() {
        // r_data = (scalar_type *) malloc( sizeof( scalar_type ) * 2*ntotal_complex ) ;
        // c_data = (data_type *) r_data;
        c_data = (data_type *) malloc( (size_t) sizeof(data_type) * ntotal_complex);
        init_Field();
        r_data = (scalar_type *) c_data;
    }
    void init_Field() {
        for (int i = 0; i < ntotal_complex; i++){
            c_data[i] = data_type(0.0,0.0);
        }
    }
    void print_values() {
        std::printf("Output array:\n");
        for (int i = 0; i < 10; i++){
            std::printf("vx[%d] %f + i%f\n", i, c_data[i].real(), c_data[i].imag());
        }
    }
    ~Field() {
        free(c_data);
    }
};*/

