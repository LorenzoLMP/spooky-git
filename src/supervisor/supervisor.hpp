#include "define_types.hpp"

#include "timer.hpp"
// use forward declarations in the header files to get around the circular dependencies
// https://stackoverflow.com/questions/994253/two-classes-that-refer-to-each-other
class Fields;
class Parameters;
class Physics;
class Timestepping;


#define LOOP 0
#define IO   1
class Supervisor {
public:
    Supervisor(int stats_frequency);

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
    void print_final_stats(int tot_steps);

    ~Supervisor();
};
