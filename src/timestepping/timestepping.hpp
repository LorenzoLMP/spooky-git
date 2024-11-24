#include "define_types.hpp"
#include <memory>
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
    TimeStepping(Supervisor &sup_in, Parameters &param);

    // Fields *fields;
    // Parameters *param;
    Supervisor *supervisor;
    std::unique_ptr<RKLegendre> rkl;
    // RKLegendre *rkl;

    int stage_step;
    unsigned int current_step;
    double current_dt;
    double dt_par;
    double dt_hyp;
    double current_time;

    data_type *d_all_scrtimestep;


    void compute_dt();
    void compute_dfield();
    void hydro_mhd_advance();
    void RungeKutta3();



    ~TimeStepping();
};
