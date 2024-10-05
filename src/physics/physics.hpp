#include "define_types.hpp"

// use forward declarations in the header files to get around the circular dependencies
// https://stackoverflow.com/questions/994253/two-classes-that-refer-to-each-other
class Fields;
class Parameters;

class Physics {
public:
    Physics();

    // Fields *fields;
    // Parameters *param;

    void Boussinesq(Fields &fields, Parameters &param);
    void AnisotropicConduction(Fields &fields, Parameters &param);
    void EntropyStratification(Fields &fields, Parameters &param);

    // int stage_step;
    // unsigned int current_step;
    // double current_dt;
    // double current_time;
    //
    // data_type *d_all_scrtimestep;

    // void compute_dt();
    // void compute_dfield();
    // void compute_dfield(Fields &fields, Parameters &param);
    // void compute_dt(Fields &fields, Parameters &param);
    // void RungeKutta3(Fields &fields, Parameters &param);

    ~Physics();
};
