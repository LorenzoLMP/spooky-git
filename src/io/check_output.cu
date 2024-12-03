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
#include "supervisor.hpp"
#include "timestepping.hpp"

void InputOutput::CheckOutput(){

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;
    std::shared_ptr<TimeStepping> timestep_ptr = supervisor_ptr->timestep_ptr;

    if( (timestep_ptr->current_time-t_lastvar)>=param_ptr->toutput_time || timestep_ptr->current_time == 0.0) {

        supervisor_ptr->timevar_timer.reset();

        fields_ptr->CleanFieldDivergence();
        std::printf("Saving output at t= %.6e \t and step n. %d \n",timestep_ptr->current_time, timestep_ptr->current_step);

        if (timestep_ptr->current_time == 0.0) {
            WriteTimevarOutputHeader();
            if (param_ptr->userOutVar.length > 0){
                WriteUserTimevarOutputHeader();
            }
        }
        if (timestep_ptr->current_time != 0.0) t_lastvar += param_ptr->toutput_time;

        WriteTimevarOutput();
        if (param_ptr->userOutVar.length > 0){
            WriteUserTimevarOutput();
        }

        supervisor_ptr->TimeIOTimevar += supervisor_ptr->timevar_timer.elapsed();
    }

    if( (timestep_ptr->current_time-t_lastsnap)>=param_ptr->toutput_flow || timestep_ptr->current_time == 0.0 ) {

        supervisor_ptr->datadump_timer.reset();


        fields_ptr->CleanFieldDivergence();
        std::printf("Starting copy back to host\n");
        fields_ptr->copy_back_to_host();
        std::printf("Saving data snap at t= %.6e \t and step n. %d \n",timestep_ptr->current_time, timestep_ptr->current_step);
        if (timestep_ptr->current_time != 0.0) t_lastsnap += param_ptr->toutput_flow;
        WriteDataFile();
        num_save++;

        supervisor_ptr->TimeIODatadump += supervisor_ptr->datadump_timer.elapsed();
    }

}
