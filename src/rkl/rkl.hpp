#include "define_types.hpp"

// use forward declarations in the header files to get around the circular dependencies
// https://stackoverflow.com/questions/994253/two-classes-that-refer-to-each-other
class Fields;
class Parameters;
class Physics;
class Supervisor;
class Timestepping;

class RKLegendre {
public:
    RKLegendre(int num, Parameters &param, Supervisor &sup);

    // Fields *fields;
    // Parameters *param;
    Supervisor *supervisor;

    // int stage_step;
    // unsigned int current_step;
    // double current_dt;
    // double current_time;

    double dt, cfl_rkl, rmax_par;

    int stage;

    data_type *d_all_dU;
    data_type *d_all_dU0;
    data_type *d_all_Uc0;
    data_type *d_all_Uc1;

    // void compute_dt();
    // void compute_dfield();
    void compute_cycle(Fields &fields, Timestepping &timestep, Physics &phys);
    // void compute_dt(Fields &fields, Parameters &param, Physics &phys);
    // void RungeKutta3(Fields &fields, Parameters &param, Physics &phys);

    ~RKLegendre();
};
