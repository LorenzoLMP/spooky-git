#include "common.hpp"

// use forward declarations in the header files to get around the circular dependencies
// https://stackoverflow.com/questions/994253/two-classes-that-refer-to-each-other
class Fields;
class Parameters;
class Supervisor;

class Physics {
public:
    Physics(Supervisor &sup_in);

    Supervisor *supervisor_ptr;

    // Fields *fields;
    // Parameters *param;

    void AdvectTemperature(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dFields);
    // void AnisotropicConduction(Fields &fields, Parameters &param);
    void AnisotropicConduction(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dTheta);

    void EntropyStratification(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dFields);

    void ParabolicTerms(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dFields);
    void ParabolicTermsSTS(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dFields);
    void HyperbolicTerms(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dFields);


    void BackgroundShear(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dFields);

    void BackgroundRotation(data_type* complex_Fields, scalar_type* real_Buffer, data_type* complex_dFields);

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
