#include "define_types.hpp"
// #include <stdlib>
#include <iostream>
#include <memory>
#include "timer.hpp"
// use forward declarations in the header files to get around the circular dependencies
// https://stackoverflow.com/questions/994253/two-classes-that-refer-to-each-other
class Fields;
class Parameters;
class Physics;
class TimeStepping;
class InputOutput;

// #include "fields.hpp"
// #include "parameters.hpp"
// #include "physics.hpp"
// #include "timestepping.hpp"
// #include "inputoutput.hpp"

#define LOOP 0
#define IO   1

class Supervisor {
public:
    Supervisor(std::string input_dir, int stats_frequency);

    std::shared_ptr<Parameters> param;
    std::shared_ptr<Fields> fields;
    std::shared_ptr<Physics> phys;
    std::shared_ptr<TimeStepping> timestep;
    std::shared_ptr<InputOutput> inout;
    // Parameters *param;
    // Fields *fields;
    // Physics *phys;
    // TimeStepping *timestep;
    // InputOutput *inout;

    Timer total_timer, timevar_timer, datadump_timer;

    int stats_frequency; // how often to print stats

    float time_delta, time_delta_2, time_delta3;
    unsigned int NumFFTs;
    float TimeSpentInFFTs;

    unsigned int AllocCpuMem;
    unsigned int AllocGpuMem;

    double TimeSpentInMainLoop;
    double TimeSpentInMainLoopPartial;
    double ElapsedWallClockTime;

    double TimeIOTimevar;
    double TimeIODatadump;

    cudaEvent_t start, stop; // for FFTs

    cudaEvent_t start_2, stop_2; // for mainloop


    void updateFFTtime();
    void updateMainLooptime();
    void print_partial_stats();
    void print_final_stats();
    void displayConfiguration();
    void executeMainLoop();
    void initialDataDump();
    void Restart(int restart_num);

    ~Supervisor();
};
