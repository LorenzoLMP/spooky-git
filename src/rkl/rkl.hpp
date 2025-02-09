#include "common.hpp"
#include <vector>

// use forward declarations in the header files to get around the circular dependencies
// https://stackoverflow.com/questions/994253/two-classes-that-refer-to-each-other
class Fields;
class Parameters;
class Physics;
class Supervisor;
class TimeStepping;

class RKLegendre {
public:
    RKLegendre(Parameters &p_in, Supervisor &sup_in);

    // Fields *fields;
    // Parameters *param;
    Supervisor *supervisor_ptr;

    std::string sts_algorithm; 	// supertimestepping algorithm. Valid choices: "sts", "rkl3"
    std::vector<int> sts_variables_index;
    std::vector<int> sts_variables_pos;
    int num_sts_vars;
    // int stage_step;
    // unsigned int current_step;
    // double current_dt;
    // double current_time;

    double dt, cfl_rkl, rmax_par;
    const int STS_MAX_STEPS = 1024;
    double *ts;
    double STS_NU = 0.01;

    int stage;
    int blocksPerGrid;

    data_type *d_all_dU, *d_all_dU0, *d_all_Uc0, *d_all_Uc1;
    data_type **d_farray_dU, **d_farray_dU0, **d_farray_Uc0, **d_farray_Uc1;

    // void compute_dt();
    // void compute_dfield();
    void compute_cycle(data_type* complex_Fields, scalar_type* real_Buffer);
    void compute_cycle_STS(data_type* complex_Fields, scalar_type* real_Buffer);
    void compute_cycle_RKL(data_type* complex_Fields, scalar_type* real_Buffer);
    // void compute_dt(Fields &fields, Parameters &param, Physics &phys);
    // void RungeKutta3(Fields &fields, Parameters &param, Physics &phys);

    ~RKLegendre();
};
