// #include "supervisor.hpp"

// use forward declarations in the header files to get around the circular dependencies
// https://stackoverflow.com/questions/994253/two-classes-that-refer-to-each-other
class Fields;
class Parameters;
class TimeStepping;
class Supervisor;
// class Timer;

class InputOutput {
public:
    InputOutput(Supervisor &sup);

    // Timer *timevar_timer, *datadump_timer;

    Supervisor *supervisor;

    double t_lastsnap; // for snapshot
    double t_lastvar;  // for volume avg/ spectral qts.
    int num_save;

    void CheckOutput(Fields &fields, Parameters &param, TimeStepping &timestep);
    void WriteDataFile(Fields &fields, Parameters &param, TimeStepping &timestep);
    void ReadDataFile(Fields &fields, Parameters &param, TimeStepping &timestep, int restart_num);
    void WriteTimevarOutput(Fields &fields, Parameters &param, TimeStepping &timestep);
    void WriteTimevarOutputHeader(Parameters &param);

    void WriteUserTimevarOutput(Fields &fields, Parameters &param, TimeStepping &timestep);
    void WriteUserTimevarOutputHeader(Parameters &param);

    ~InputOutput();
};
