// #include "supervisor.hpp"
#include "common.hpp"
// use forward declarations in the header files to get around the circular dependencies
// https://stackoverflow.com/questions/994253/two-classes-that-refer-to-each-other
class Fields;
class Parameters;
class TimeStepping;
class Supervisor;
// class Timer;

class InputOutput {
public:
    InputOutput(Supervisor &sup_in);

    // Timer *timevar_timer, *datadump_timer;

    Supervisor *supervisor_ptr;

    double t_lastsnap; // for snapshot
    double t_lastvar;  // for volume avg/ spectral qts.
    int num_save;

    void CheckOutput();
    void WriteDataFile();
    void ReadDataFile(int restart_num);

    void WriteTimevarOutput();
    void WriteTimevarOutputHeader();

    void WriteUserTimevarOutput();
    void WriteUserTimevarOutputHeader();

    void UnshearOutput(data_type *AllComplexFields, scalar_type *AllFieldsRealTmp, double current_time);
    void UnshearField(data_type *vecField, double current_time);

    ~InputOutput();
};
