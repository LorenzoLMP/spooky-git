#include "define_types.hpp"
// #include "fields.hpp"
// #include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
// #include "../libs/HighFive/include/highfive/highfive.hpp"
// #include <highfive/highfive.hpp>
#include "inputoutput.hpp"
#include "fields.hpp"
#include "hdf5_io.hpp"
// #include "output_timevar.hpp"
#include "parameters.hpp"

#include "timestepping.hpp"

void InputOutput::CheckOutput(Fields &fields, Parameters &param, TimeStepping &timestep){

    if( (timestep.current_time-t_lastvar)>=param.toutput_time || timestep.current_time == 0.0) {
        fields.CleanFieldDivergence();
        std::printf("Saving output at t= %.6e \t and step n. %d \n",timestep.current_time, timestep.current_step);

        if (timestep.current_time == 0.0) {
            WriteTimevarOutputHeader(param);
            if (param.userOutVar.length > 0){
                WriteUserTimevarOutputHeader(param);
            }
        }
        if (timestep.current_time != 0.0) t_lastvar += param.toutput_time;

        WriteTimevarOutput(fields, param, timestep);
        if (param.userOutVar.length > 0){
            WriteUserTimevarOutput(fields, param, timestep);
        }
    }

    if( (timestep.current_time-t_lastsnap)>=param.toutput_flow || timestep.current_time == 0.0 ) {
        fields.CleanFieldDivergence();
        std::printf("Starting copy back to host\n");
        fields.copy_back_to_host();
        std::printf("Saving data snap at t= %.6e \t and step n. %d \n",timestep.current_time, timestep.current_step);
        if (timestep.current_time != 0.0) t_lastsnap += param.toutput_flow;
        WriteDataFile(fields, param, timestep);
        num_save++;
    }

}
