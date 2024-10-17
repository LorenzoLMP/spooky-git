#include "define_types.hpp"

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
    Supervisor();

    float time_delta, time_delta_2;
    unsigned int NumFFTs;
    double TimeSpentInFFTs;

    unsigned int AllocCpuMem;
    unsigned int AllocGpuMem;

    double TimeSpentInMainLoop;
    double ElapsedWallClockTime;

    cudaEvent_t start, stop; // for FFTs

    cudaEvent_t start_2, stop_2; // for mainloop


    void updateFFTtime();
    void updateMainLooptime();


    ~Supervisor();
};
