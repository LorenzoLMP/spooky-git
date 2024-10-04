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

    if( (timestep.current_time-t_lastsnap)>=param.toutput_flow) {
        // ComputeDivergence();
        fields.CleanFieldDivergence();
        // ComputeDivergence();
        std::printf("Starting copy back to host\n");
        fields.copy_back_to_host();
        std::printf("Saving data snap at t= %.6e \t and step n. %d \n",timestep.current_time, timestep.current_step);
        // std::printf("Saving data file at t= %.6e \n",current_time);
        write_data_file(fields, param, timestep);
        t_lastsnap += param.toutput_flow;
        num_save++;
    }

    if( (timestep.current_time-t_lastvar)>=param.toutput_time) {
        // ComputeDivergence();
        fields.CleanFieldDivergence();
        // ComputeDivergence();
        // std::printf("Starting copy back to host\n");
        // copy_back_to_host();
        std::printf("Saving output at t= %.6e \t and step n. %d \n",timestep.current_time, timestep.current_step);
        // std::printf("Saving data file at step n. %d \n",current_step);
        // std::printf("Saving data file at t= %.6e \n",current_time);
        write_data_output(fields, param, timestep);
        t_lastvar += param.toutput_time;
        // num_save++;
    }
}
