#include "define_types.hpp"
// #include "fields.hpp"
#include "cufft_routines.hpp"
#include "spooky.hpp"
#include "common.hpp"
// #include "../libs/HighFive/include/highfive/highfive.hpp"
#include <highfive/highfive.hpp>
#include "fields.hpp"
#include "hdf5_io.hpp"
#include "parameters.hpp"
#include "inputoutput.hpp"
#include "timestepping.hpp"

using namespace HighFive;


void InputOutput::WriteDataFile(Fields &fields, Parameters &param, TimeStepping &timestep) {

    // NVTX3_FUNC_RANGE();
#ifdef DEBUG
    std::printf("////////////////////////// \n");
    std::printf("Writing data file... \n");
    std::printf("////////////////////////// \n");
#endif


    // double t0        = param.t_initial;
    // double time_save = timestep.current_time;
    // double tend     = param.t_final;
    std::printf("t0: %.2e \t time_save: %.2e \t tend: %.2e \n",param.t_initial,timestep.current_time,param.t_final);

    char data_snap_name[16];
    std::sprintf(data_snap_name,"snap%04i.h5",num_save);
    std::string fname = param.output_dir + std::string("/data/") + std::string(data_snap_name);

    std::cout << fname << std::endl;
    std::printf("-----------------Writing snap %04i at time_save: %.2e \n",num_save,timestep.current_time);
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
    dataset.write(param.t_initial);

    dataset = file.createDataSet<scalar_type>("t_save", data_scalar);
    dataset.write(timestep.current_time);

    dataset = file.createDataSet<scalar_type>("t_end", data_scalar);
    dataset.write(param.t_final);

    dataset = file.createDataSet<scalar_type>("t_lastsnap", data_scalar);
    dataset.write(t_lastsnap);

    dataset = file.createDataSet<scalar_type>("t_lastvar", data_scalar);
    dataset.write(t_lastvar);

    dataset = file.createDataSet<int>("step", data_scalar);
    dataset.write(timestep.current_step);

    std::printf("t_lastsnap: %.2e \t t_lastvar: %.2e \n",t_lastsnap,t_lastvar);



    long int idx, idx_complex;
    scalar_type *scratch;
    scratch = (scalar_type *) malloc( (size_t) sizeof(scalar_type) * ntotal);
    DataSpace data_field(ntotal);

    for (int n = 0; n < fields.num_fields; n++){

        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){
                for (int k = 0; k < nz; k++){
                    // need to rearrange the data to remove zero padding
                    idx = k + nz * ( j + i * ny);
                    idx_complex = k + (nz/2+1)*2 * ( j + i * ny);
                    // idx_complex = k + (nz + 2) * j + (nz + 2) * ny * i;
                    // scratch[idx] = 1.0;
                    scratch[idx] = fields.farray_r[n][idx_complex];
                }
            }
        }
        std::string var_name(list_of_variables[n]);
        dataset = file.createDataSet<scalar_type>(var_name, data_field);
        dataset.write(scratch);

    }

    free(scratch);
    // std::printf("Finished writing data file.\n");
}


void InputOutput::ReadDataFile(Fields &fields, Parameters &param, TimeStepping &timestep, int restart_num) {

    // NVTX3_FUNC_RANGE();
#ifdef DEBUG
    std::printf("Reading data file... \n");
#endif

    char data_snap_name[16];
    double time_save;

    long int idx, idx_complex;
    unsigned int save_step;
    scalar_type *scratch;
    scratch = (scalar_type *) malloc( (size_t) sizeof(scalar_type) * ntotal);

    // DataSet dataset;
    DataSpace data_scalar(1);
    DataSpace data_field(ntotal);

    /***************************
     * Select the right data file OR
     * find the most recent data file
     * **************************/

    if (restart_num == -1){
        // find the most recent data file
    }
    else {
        std::sprintf(data_snap_name,"snap%04i.h5",restart_num);
    }


    std::string fname = param.output_dir + std::string("/data/") + std::string(data_snap_name);

    // std::printf("Reading from data file %s \n",fname);
    std::cout << "Reading from data file \t" << fname << std::endl;

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

        // std::printf("time_save: %.2f \t tend: %.2f \n",param.t_initial,timestep.current_time,param.t_final);
        std::printf("t_lastsnap: %.2e \t t_lastvar: %.2e \n",t_lastsnap,t_lastvar);

        num_save = restart_num+1;
        timestep.current_time = time_save;
        timestep.current_step = save_step;


        std::printf("-----------------Reading from snap %04i at time_save: %.2e \n",restart_num,time_save);


        // Now read data_fields
        for (int n = 0; n < fields.num_fields; n++){

            std::string var_name(list_of_variables[n]);
            dataset = file.getDataSet(var_name);
            //
            // let's read our vector into the scratch
            dataset.read(scratch);

            // let's rearrange it to remove zero padding
            for (int i = 0; i < nx; i++){
                for (int j = 0; j < ny; j++){
                    for (int k = 0; k < nz; k++){
                        // two indices
                        idx = k + nz * ( j + i * ny);
                        idx_complex = k + (nz/2+1)*2 * ( j + i * ny);

                        fields.farray_r[n][idx_complex] = scratch[idx];
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
        std::printf("!!!! Error: restart file not found, starting a new run from t=0.00.\n");
        std::printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        param.restart = 0;

    }

    free(scratch);

}
