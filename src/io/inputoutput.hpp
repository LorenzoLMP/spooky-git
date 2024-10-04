




// use forward declarations in the header files to get around the circular dependencies
// https://stackoverflow.com/questions/994253/two-classes-that-refer-to-each-other
class Fields;
class Parameters;
class TimeStepping;

class InputOutput {
public:
    InputOutput();

    // Fields *fields;
    // TimeStepping *timestep;
    // Parameters *param;

    double t_lastsnap; // for snapshot
    double t_lastvar;  // for volume avg/ spectral qts.
    int num_save;

    void CheckOutput(Fields &fields, Parameters &param, TimeStepping &timestep);
    void write_data_file(Fields &fields, Parameters &param, TimeStepping &timestep);
    void write_data_output(Fields &fields, Parameters &param, TimeStepping &timestep);
    void write_data_output_header(Parameters &param);

    ~InputOutput();
};
