#include "../define_types.hpp"
// #include "fields.hpp"
#include "../cufft_routines.hpp"
#include "../spooky.hpp"
#include "../common.hpp"
// #include "../libs/HighFive/include/highfive/highfive.hpp"
#include <highfive/highfive.hpp>
#include "../fields.hpp"
#include "hdf5_io.hpp"

using namespace HighFive;

// const std::string file_name("dataset.h5");
// const std::string dataset_name("dset");

// void Fields::CheckOutput(){
//
//     if( (current_time-t_lastsnap)>=param->toutput_flow) {
//         // ComputeDivergence();
//         CleanFieldDivergence();
//         // ComputeDivergence();
//         std::printf("Starting copy back to host\n");
//         copy_back_to_host();
//         std::printf("Saving data file at step n. %d \n",current_step);
//         std::printf("Saving data file at t= %.6e \n",current_time);
//         write_data_file();
//         t_lastsnap += param->toutput_flow;
//         num_save++;
//     }
// }

void Fields::write_data_file() {

    NVTX3_FUNC_RANGE();
    std::printf("Writing data file... \n");

    double t0        = param->t_initial;
    double time_save = current_time;
    double tend     = param->t_final;
    // double times[3] = {param->t_initial, current_dt, param->t_final};
    // char file_name[256];
    // sprintf(filename,"%s/data/v%04i.vtk",param->output_dir,num_save);
    // std::sprintf(file_name,"%s/data/snap%04i.h5",param->output_dir,num_save);

    char data_snap_name[16];
    std::sprintf(data_snap_name,"snap%04i.h5",num_save);
    std::string fname = param->output_dir + std::string("/data/") + std::string(data_snap_name);
    // we are assuming that the fields have been copied back to cpu and are real
    // we create a new hdf5 file
    // std::string fname(file_name);
    // File file();
    File file(fname, File::ReadWrite | File::Create | File::Truncate);
    // try {
    // file.getName();
    // }
    // catch (const std::exception& err) {
    // std::cerr << err.what() << std::endl;
    // // std::cerr << program;
    // std::exit(1);
    // }
    // File file(fname, File::ReadWrite | File::Create | File::Truncate);


    DataSpace data_scalar(1);
    DataSet dataset = file.createDataSet<scalar_type>("t_start", data_scalar);
    dataset.write(t0);



    dataset = file.createDataSet<scalar_type>("t_save", data_scalar);
    dataset.write(time_save);

    dataset = file.createDataSet<scalar_type>("t_end", data_scalar);
    dataset.write(tend);


    //
    // std::printf("About to write scratch\n");
    long int idx, idx_complex;
    scalar_type *scratch;
    scratch = (scalar_type *) malloc( (size_t) sizeof(scalar_type) * ntotal);
    DataSpace data_field(ntotal);

    for (int n = 0; n < num_fields; n++){

        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){
                for (int k = 0; k < nz; k++){
                    // need to rearrange the data to remove zero padding
                    idx = k + nz * ( j + i * ny);
                    idx_complex = k + (nz/2+1)*2 * ( j + i * ny);
                    // idx_complex = k + (nz + 2) * j + (nz + 2) * ny * i;
                    // scratch[idx] = 1.0;
                    scratch[idx] = farray_r[n][idx_complex];
                }
            }
        }
        // std::printf("Finished writing scratch\n");
        // // let's create a dataset of native integer with the size of the vector
        // // 'data'

        std::string var_name(list_of_variables[n]);
        dataset = file.createDataSet<scalar_type>(var_name, data_field);
        //
        // // let's write our vector of int to the HDF5 dataset
        dataset.write(scratch);

    }

    free(scratch);
    // std::printf("Finished writing data file.\n");
}
