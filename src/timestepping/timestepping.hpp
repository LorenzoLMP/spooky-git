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
    Supervisor *supervisor_ptr;
    std::unique_ptr<RKLegendre> rkl;
    // RKLegendre *rkl;

    int stage_step;
    unsigned int current_step;
    double current_dt;
    double dt_par;
    double dt_hyp;
    double current_time;

    data_type *d_all_scrtimestep;


    void compute_dt(data_type* complexFields, scalar_type* realBuffer);
    void compute_dfield(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dFields);
    void hydro_mhd_advance(Fields* fields_ptr);
    void RungeKutta3(data_type* complex_Fields, scalar_type* real_Buffer);



    ~TimeStepping();
};
