#include "common.hpp"
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
    Supervisor();

    std::shared_ptr<Parameters> param_ptr;
    std::shared_ptr<Fields> fields_ptr;
    std::shared_ptr<Physics> phys_ptr;
    std::shared_ptr<TimeStepping> timestep_ptr;
    std::shared_ptr<InputOutput> inout_ptr;
    // Parameters *param;
    // Fields *fields;
    // Physics *phys;
    // TimeStepping *timestep;
    // InputOutput *inout;

    Timer total_timer, timevar_timer, datadump_timer;

    int stats_frequency; // how often to print stats (in steps)

    float time_delta, time_delta_2, time_delta3;
    unsigned int NumFFTs;
    float TimeSpentInFFTs;

    // in bytes
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
    void Restart();

    void Complex2RealFields(data_type* ComplexField_in, int num_fields);
    void Complex2RealFields(data_type* ComplexField_in, scalar_type* RealField_out, int num_fields);

    template <typename T>
    void spookyGpuAlloc(T** devPtr, size_t size){
        CUDA_RT_CALL(cudaMalloc(devPtr, size));
        AllocGpuMem += size;
    };

    ~Supervisor();
};
