#include "define_types.hpp"
// #include "wavevector.hpp"
// #include "parameters.hpp"


// use forward declarations in the header files to get around the circular dependencies
// https://stackoverflow.com/questions/994253/two-classes-that-refer-to-each-other

class Parameters;
class TimeStepping;
class Supervisor;

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

    Wavevector(Parameters &p_in);
    // Wavevector();

    // void init_Wavevector(Parameters *p_in);
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
    // int num_fields;
    int num_tmp_array;
    Supervisor *supervisor_ptr;
    // double current_dt;
    // double current_time;
    // int stage_step;
    // double t_lastsnap; // for snapshot
    // double t_lastvar;  // for volume avg/ spectral qts.
    // int num_save;
    // unsigned int current_step;
    data_type *all_fields, *all_dfields;
    data_type **farray, **dfarray;
    scalar_type **farray_r, **dfarray_r;
 
    data_type *d_all_fields, *d_all_dfields, *d_all_tmparray;
    data_type **d_farray, **d_dfarray, **d_tmparray;
    scalar_type **d_farray_r, **d_dfarray_r, **d_tmparray_r;

    scalar_type *d_all_buffer_r;
    scalar_type **d_farray_buffer_r;
    
    Wavevector wavevector;
    // Parameters *param;
    // TimeStepping *timestep;
    // Fields(Parameters *p_in, TimeStepping *timestep_in, int num);
    Fields(Supervisor &sup_in, Parameters &p_in);
    // void init_Fields( int num, Parameters *p_in);
    void initSpatialStructure();
    void print_host_values();
    void print_device_values();
    void allocate_and_move_to_gpu();
    void compare_with_original();
    // void do_operations(Wavevector *wavevector);
    // void do_operations();
    // void do_multiplications();
    // double advance_timestep( double t, double t_end, int* p_step);
    // void compute_dfield(int stage_step);
    // void compute_dt();
    void copy_back_to_host();

    // void CheckOutput();
    // void write_data_file();
    // void write_data_output();
    // void write_data_output_header();

    void clean_gpu();
    // void RungeKutta3();
    double ComputeDivergence(data_type* complex_Fields);
    void CleanFieldDivergence();
    void CheckSymmetries();
    // void CleanDivergence();

    // void Complex2RealFields(data_type* ComplexField_in, scalar_type* RealField_out, int num_fields);

    // void Boussinesq();
    // void AnisotropicConduction();
    // void EntropyStratification();

    ~Fields();
};


