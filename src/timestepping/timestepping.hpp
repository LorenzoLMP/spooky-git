#include "define_types.hpp"
// #include "rkl.hpp"

// use forward declarations in the header files to get around the circular dependencies
// https://stackoverflow.com/questions/994253/two-classes-that-refer-to-each-other
class Fields;
class Parameters;
class Physics;
class Supervisor;
class RKLegendre;

class TimeStepping {
public:
    TimeStepping(int num, Parameters &param, Supervisor &sup);

    // Fields *fields;
    // Parameters *param;
    Supervisor *supervisor;
    RKLegendre *rkl;

    int stage_step;
    unsigned int current_step;
    double current_dt;
    double dt_par;
    double dt_hyp;
    double current_time;

    data_type *d_all_scrtimestep;

    // void compute_dt();
    // void compute_dfield();
    void compute_dfield(Fields &fields, Parameters &param, Physics &phys);
    void compute_dt(Fields &fields, Parameters &param, Physics &phys);
    void RungeKutta3(Fields &fields, Parameters &param, Physics &phys);

    ~TimeStepping();
};
