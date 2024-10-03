#include "define_types.hpp"
// #include "wavevector.hpp"
// #include "parameters.hpp"


// use forward declarations in the header files to get around the circular dependencies
// https://stackoverflow.com/questions/994253/two-classes-that-refer-to-each-other

class Parameters;

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

    Wavevector(scalar_type Lx, scalar_type Ly, scalar_type Lz);

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
    int stage_step;
    double t_lastsnap; // for snapshot
    double t_lastvar;  // for volume avg/ spectral qts.
    int num_save;
    unsigned int current_step;
    data_type *all_fields, *all_dfields;
    data_type **farray, **dfarray;
    scalar_type **farray_r, **dfarray_r;
 
    data_type *d_all_fields, *d_all_dfields, *d_all_scrtimestep, *d_all_tmparray;
    data_type **d_farray, **d_dfarray, **d_tmparray;
    scalar_type **d_farray_r, **d_dfarray_r, **d_tmparray_r;
    
    Wavevector wavevector;
    Parameters *param;
    Fields( int num, Parameters *p_in );
    void init_Fields();
    void init_SpatialStructure();
    void print_host_values();
    void print_device_values();
    void allocate_and_move_to_gpu();
    void compare_with_original();
    // void do_operations(Wavevector *wavevector);
    // void do_operations();
    // void do_multiplications();
    // double advance_timestep( double t, double t_end, int* p_step);
    void compute_dfield();
    void compute_dt();
    void copy_back_to_host();

    void CheckOutput();
    void write_data_file();
    void write_data_output();
    void write_data_output_header();

    void clean_gpu();
    void RungeKutta3();
    void ComputeDivergence();
    void CleanFieldDivergence( );
    void CheckSymmetries();

    void Boussinesq();
    void AnisotropicConduction();
    void EntropyStratification();

    ~Fields();
};


