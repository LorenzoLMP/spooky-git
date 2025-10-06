#include "common.hpp"
#include <filesystem>
// #include <stdlib.h>
// #include "fields.hpp"
#include "cufft_routines.hpp"
// #include "../libs/HighFive/include/highfive/highfive.hpp"
#include <highfive/highfive.hpp>
#include "fields.hpp"
// #include "hdf5_io.hpp"
#include "parameters.hpp"
#include "inputoutput.hpp"
#include "timestepping.hpp"
#include "supervisor.hpp"

using namespace HighFive;

int fileCounter(std::string dir);

void InputOutput::WriteDataFile() {

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;
    std::shared_ptr<TimeStepping> timestep_ptr = supervisor_ptr->timestep_ptr;

    // NVTX3_FUNC_RANGE();

    if (param_ptr->debug > 0){
        std::printf("////////////////////////// \n");
        std::printf("Writing data file... \n");
        std::printf("////////////////////////// \n");
    }


    // double t0        = param_ptr->t_initial;
    // double time_save = timestep_ptr->current_time;
    // double tend     = param_ptr->t_final;
    std::printf("t0: %.2e \t time_save: %.2e \t tend: %.2e \n",param_ptr->t_initial,timestep_ptr->current_time,param_ptr->t_final);

    char data_snap_name[16];
    std::sprintf(data_snap_name,"snap%04i.h5",num_save);
    std::string fname = param_ptr->output_dir + std::string("/data/") + std::string(data_snap_name);

    std::cout << fname << std::endl;
    std::printf("-----------------Writing snap %04i at time_save: %.2e \n",num_save,timestep_ptr->current_time);
    // we are assuming that the fields have been copied back to cpu and are real
    // we create a new hdf5 file
    File file(fname, File::ReadWrite | File::Create | File::Truncate);

    // try {
    // file.getName();
    // }
    // catch (const std::exception& err) {
    // std::cerr << err.what() << std::endl;
    // // std::cerr << program;
    // std::exit(1);
    // }

    // DataSet dataset;
    DataSpace data_scalar(1);
    DataSet dataset = file.createDataSet<scalar_type>("t_start", data_scalar);
    dataset.write(param_ptr->t_initial);

    dataset = file.createDataSet<scalar_type>("t_save", data_scalar);
    dataset.write(timestep_ptr->current_time);

    // dataset = file.createDataSet<scalar_type>("t_end", data_scalar);
    // dataset.write(param_ptr->t_final);

    dataset = file.createDataSet<scalar_type>("t_lastsnap", data_scalar);
    dataset.write(t_lastsnap);

    dataset = file.createDataSet<scalar_type>("t_lastvar", data_scalar);
    dataset.write(t_lastvar);

    dataset = file.createDataSet<int>("step", data_scalar);
    dataset.write(timestep_ptr->current_step);

    std::printf("t_lastsnap: %.2e \t t_lastvar: %.2e \n",t_lastsnap,t_lastvar);



    long int idx, idx_complex;
    scalar_type *scratch;
    scratch = (scalar_type *) malloc( (size_t) sizeof(scalar_type) * grid.NTOTAL);
    DataSpace data_field(grid.NTOTAL);

    for (int n = 0; n < vars.NUM_FIELDS; n++){

        for (int i = 0; i < grid.NX; i++){
            for (int j = 0; j < grid.NY; j++){
                for (int k = 0; k < grid.NZ; k++){
                    // need to rearrange the data to remove zero padding
                    idx = k + grid.NZ * ( j + i * grid.NY);
                    idx_complex = k + (grid.NZ/2+1)*2 * ( j + i * grid.NY);
                    // idx_complex = k + (grid.NZ + 2) * j + (grid.NZ + 2) * grid.NY * i;
                    // scratch[idx] = 1.0;
                    scratch[idx] = fields_ptr->farray_r[n][idx_complex];
                }
            }
        }


        std::string var_name(vars.VAR_LIST[n]);
        dataset = file.createDataSet<scalar_type>(var_name, data_field);
        dataset.write(scratch);

    }

    free(scratch);
    // std::printf("Finished writing data file.\n");
}


void InputOutput::ReadDataFile(int restart_num) {

    std::shared_ptr<Fields> fields_ptr = supervisor_ptr->fields_ptr;
    std::shared_ptr<Parameters> param_ptr = supervisor_ptr->param_ptr;
    std::shared_ptr<TimeStepping> timestep_ptr = supervisor_ptr->timestep_ptr;

    // NVTX3_FUNC_RANGE();
    if (param_ptr->debug > 0){
        std::printf("Reading data file... \n");
    }

    char data_snap_name[16];
    double time_save;
    int restart_tmp;
    long int idx, idx_complex;
    unsigned int save_step;
    scalar_type *scratch;
    scratch = (scalar_type *) malloc( (size_t) sizeof(scalar_type) * grid.NTOTAL);

    // DataSet dataset;
    DataSpace data_scalar(1);
    DataSpace data_field(grid.NTOTAL);

    /***************************
     * Select the right data file OR
     * find the most recent data file
     * **************************/

    // find the most recent data file
    int most_recent_snap = fileCounter(param_ptr->output_dir + std::string("/data/"));
    std::printf("most recent snap: %d \n", most_recent_snap);

    if (most_recent_snap >= 0) { // some data files exist!
        if (restart_num > most_recent_snap){
            std::printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
            std::printf("!!!! Error: restart file not found, restarting from last available data file...\n");
            std::printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
            restart_tmp = most_recent_snap;
        }
        else if (restart_num == -1){
            std::printf("Restarting from last available data file...\n");
            restart_tmp = most_recent_snap;
        }
        else {
            std::cout << "Data file selected for restarting: \t" << restart_num << std::endl;
            restart_tmp = restart_num;
        }
        std::sprintf(data_snap_name,"snap%04i.h5",restart_tmp);

        std::string fname = param_ptr->output_dir + std::string("/data/") + std::string(data_snap_name);

        // std::printf("Reading from data file %s \n",fname);
        std::cout << "Data file selected for restarting: \t" << fname << std::endl;

        // sanity check
        if (std::filesystem::exists(fname)){
            std::cout << "The data file requested exists. Attempting to read data from it..." << std::endl;
        }
        else{
            std::printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
            std::printf("!!!! Error: restart file not found, check that you are choosing the right file.\n");
            std::printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
            // return;
            exit(0);
        }

        try {
            File file(fname, File::ReadOnly);

            DataSet dataset = file.getDataSet("t_save");
            dataset.read(time_save);

            dataset = file.getDataSet("t_lastsnap");
            dataset.read(t_lastsnap);

            dataset = file.getDataSet("t_lastvar");
            dataset.read(t_lastvar);

            dataset = file.getDataSet("step");
            dataset.read(save_step);

            // std::printf("time_save: %.2f \t tend: %.2f \n",param_ptr->t_initial,timestep_ptr->current_time,param_ptr->t_final);
            std::printf("t_lastsnap: %.2e \t t_lastvar: %.2e \n",t_lastsnap,t_lastvar);

            num_save = restart_tmp+1;
            timestep_ptr->current_time = time_save;
            timestep_ptr->current_step = save_step;


            std::printf("-----------------Reading from snap %04i at time_save: %.2e \n",restart_tmp,time_save);


            // Now read data_fields
            for (int n = 0; n < vars.NUM_FIELDS; n++){

                std::string var_name(vars.VAR_LIST[n]);
                dataset = file.getDataSet(var_name);
                //
                // let's read our vector into the scratch
                dataset.read(scratch);

                // let's rearrange it to remove zero padding
                for (int i = 0; i < grid.NX; i++){
                    for (int j = 0; j < grid.NY; j++){
                        for (int k = 0; k < grid.NZ; k++){
                            // two indices
                            idx = k + grid.NZ * ( j + i * grid.NY);
                            idx_complex = k + (grid.NZ/2+1)*2 * ( j + i * grid.NY);

                            fields_ptr->farray_r[n][idx_complex] = scratch[idx];
                            // if (i < 10 && k == 0) std::printf("farray_r %.2e \n",scratch[idx]);
                        }
                    }
                }
            }
            std::printf("Finished reading data file.\n");

        } catch (const Exception& err) {
            // std::cerr << err.what() << std::endl;
            // std::exit(1);
            std::printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
            std::printf("!!!! Error: could not read from data file. \n");
            std::printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
            // param_ptr->restart = 0;
            exit(0);

        }

    }
    else { // no datafiles found
        std::printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        std::printf("!!!! Error: no restart files found, starting a new run...\n");
        std::printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        param_ptr->restart = 0;
    }

    free(scratch);

}

// from https://stackoverflow.com/a/36878471
int fileCounter(std::string dir){
    int returnedCount = -1;
    int possibleMax = 5000000; //some number you can expect.
    bool status;
    char data_snap_name[16];

    for (int istarter = 0; istarter < possibleMax; istarter++){
        std::string fileName = "";
        fileName.append(dir);
        // fileName.append(prefix);
        std::sprintf(data_snap_name,"snap%04i.h5",istarter);
        // fileName.append(to_string(istarter));
        fileName.append(data_snap_name);
        status = std::filesystem::exists(fileName);

        if (!status)
            break;

        returnedCount++;
    }
    return returnedCount;
}
