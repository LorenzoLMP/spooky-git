#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
// #include "libconfig/libconfig.h++"
#include <libconfig.h>
// #include "common.hpp"
#include "parameters.hpp"
// #include "spooky_outputs.hpp"

#define SPOOKY_CONFIG_FILENAME		"spooky.cfg"
// #define CONFIG_FILENAME		"./spooky.cfg"


// SpookyOutput::SpookyOutput() {
//     // double lx, ly, lz;
//     // read_Parameters();
// }
//
// SpookyOutput::~SpookyOutput() {
// }

// Parameters::Parameters() : spookyOutVar(), userOutVar() {
	// SpookyOutput *spookyOutVar();
	// UserOutput *userOutVar();
    // double lx, ly, lz;
    // read_Parameters();
	// field = field_in;
// }

Parameters::~Parameters() {
}

// void Parameters::add

// void Parameters::read_Parameters(std::string input_dir) {
Parameters::Parameters(std::string input_dir) : spookyOutVar(), userOutVar() {
// void Parameters::read_Parameters() {
	// field = fields_in;
	// Read the config file and initialize everyting
	config_t	config;		// Initialize the structure
	config_setting_t * setting;	// a setting structure
	int tmp_v;
	int i,n;
	std::string config_fullpath(input_dir);
	config_fullpath.append("/");
	config_fullpath.append(std::string(SPOOKY_CONFIG_FILENAME));

	const char * CONFIG_FILENAME = config_fullpath.c_str();

	const char * configname;

	const char * temp_string;

    const char * temp_output;

	// SpookyOutput spookyOutVar;
	// DEBUG_START_FUNC;

	// if(rank==0) {
		config_init(&config);

		if(!config_read_file(&config, CONFIG_FILENAME)) {
			std::printf("Error reading configuration file in line %d: %s\n", config_error_line(&config), config_error_text(&config));
			// ERROR_HANDLER(ERROR_CRITICAL, "Failed to read the configuration file");
		}

		if(config_lookup_string(&config,"configname",&configname)) {
			std::printf("Using config file: %s at %s.\n",configname,CONFIG_FILENAME);
		}
		// read physics parameters-------------------------------------------------------------------------------
		// std::printf("Before reading boxsizes\n");
		if(!config_lookup_float(&config, "physics.boxsize.[0]",&lx)) {
			lx = 1.0;
		}
		if(!config_lookup_float(&config, "physics.boxsize.[1]",&ly)) {
			ly = 1.0;
		}
		if(!config_lookup_float(&config, "physics.boxsize.[2]",&lz)) {
			lz = 1.0;
		}
		// std::printf("Read boxsizes\n");

		if(!config_lookup_float(&config, "physics.reynolds",&reynolds)) {
			reynolds = 1.0;
		}
		nu = 1./reynolds;
		if(!config_lookup_float(&config, "physics.reynolds_magnetic",&reynolds_m)) {
			reynolds_m = 1.0;
		}
		nu_m = 1./reynolds_m;
		if(!config_lookup_float(&config, "physics.reynolds_thermic",&reynolds_th)) {
			reynolds_th = 1.0;
		}
		nu_th = 1./reynolds_th;
		if(!config_lookup_float(&config, "physics.reynolds_anisotropic",&reynolds_ani)) {
			reynolds_ani = 1.0;
		}

		if(!config_lookup_float(&config, "physics.reynolds_Braginskii",&reynolds_B)) {
			reynolds_B = 1.0;
		}

		// if(!config_lookup_float(&config, "physics.x_Hall",&x_hall)) {
		// 	x_hall = 1.0;
		// }

		if(!config_lookup_float(&config, "physics.brunt_vaissala_squared",&N2)) {
			N2 = 0.0;
		}

		if(!config_lookup_float(&config, "physics.mti_frequency_squared",&OmegaT2)) {
			OmegaT2 = 0.0;
		}

		if(!config_lookup_float(&config, "physics.conductivity_alpha",&ALPHA)) {
			ALPHA = 0.0;
		}

		// if(!config_lookup_float(&config, "physics.omega",&omega)) {
		// 	omega = 0.0;
		// }
// #ifndef WITH_ROTATION
// 		// Omega should be forced to zero in order to be fool-proof
// 		omega = 0.0;
// #endif
// 		if(!config_lookup_float(&config, "physics.shear",&shear)) {
// 			shear = 0.0;
// 		}
// #ifndef WITH_SHEAR
// 		// same for the shear
// 		shear = 0.0;
// #endif
// 		if(!config_lookup_float(&config, "physics.omega_shear",&omega_shear)) {
// 			omega_shear = 0.0;
// 		}

		if(!config_lookup_float(&config, "physics.sound_speed",&cs)) {
			cs = 1.0;
		}

		// Particles parameters-------------------------------------------------------------------------------------
		// if(!config_lookup_int(&config, "particles.n",&tmp_v)) {
		// 	particles_n = 1000;
		// }
		// else {
		// 	particles_n = (int) tmp_v;
		// }
  //
		// if(!config_lookup_float(&config, "particles.mass",&particles_mass)) {
		// 	particles_mass = 1.0;
		// }
  //
		// if(!config_lookup_float(&config, "particles.stime",&particles_stime)) {
		// 	particles_stime = 1.0;
		// }
  //
		// if(!config_lookup_float(&config, "particles.dg_ratio",&particles_dg_ratio)) {
		// 	particles_dg_ratio = 0.01;
		// }
  //
		// if(!config_lookup_float(&config, "particles.epsilon",&particles_epsilon)) {
		// 	particles_epsilon = 0.1;
		// }

		// Code parameters-------------------------------------------------------------------------------------

		if(!config_lookup_float(&config, "code.cfl",&cfl)) {
			cfl = 1.5;
		}
		if(!config_lookup_float(&config, "code.cfl_par",&cfl_par)) {
			cfl_par = 1.0;
		}
		if(!config_lookup_float(&config, "code.safety_sts",&safety_sts)) {
			safety_sts = 50.0;
		}
		if(!config_lookup_float(&config, "code.safety_source",&safety_source)) {
			safety_source = 0.2;
		}
		if(!config_lookup_float(&config, "code.t_initial",&t_initial)) {
			t_initial = 0.0;
		}
		if(!config_lookup_float(&config, "code.t_final",&t_final)) {
			t_final = 1.0;
		}
		if(!config_lookup_float(&config, "code.max_t_elapsed",&max_t_elapsed)) {
			max_t_elapsed = 1e30;
		}
		if(!config_lookup_int(&config, "code.interface_check",&tmp_v)) {
			interface_check = 5;
		}
		else {
			interface_check = (int) tmp_v;
		}
		if(!config_lookup_bool(&config, "code.interface_output_file",&interface_output_file)) {
			interface_output_file = 0;
		}
		if(!config_lookup_bool(&config, "code.force_symmetries",&force_symmetries)) {
			force_symmetries = 0;
		}
		if(!config_lookup_int(&config, "code.symmetries_step",&tmp_v)) {
			symmetries_step = 20;
		}
		else {
			symmetries_step = (int) tmp_v;
		}
		if(!config_lookup_bool(&config, "code.antialiasing",&antialiasing)) {
			antialiasing = 1;
		}
		if(!config_lookup_bool(&config, "code.restart",&restart)) {
			restart = 0;
		}

		// Output parameters-------------------------------------------------------------------------------------
		if(!config_lookup_float(&config, "output.timevar_step",&toutput_time)) {
			toutput_time = 1.0;
		}
		// if(!config_lookup_float(&config, "output.spectrum_step",&toutput_spec)) {
		// 	toutput_spec = 1.0;
		// }
		if(!config_lookup_float(&config, "output.snapshot_step",&toutput_flow)) {
			toutput_flow = 1.0;
		}
		if(!config_lookup_float(&config, "output.dump_step",&toutput_dump)) {
			toutput_dump = 1.0;
		}
		if(!config_lookup_bool(&config, "output.vorticity",&output_vorticity)) {
			output_vorticity = 0;
		}
		if(config_lookup_string(&config, "output.output_dir",&temp_output)) {
			std::printf("output dir: %s.\n",temp_output);
			// char temp_dest[256];
			// strcpy(output_dir, temp_output);
			output_dir = std::string(temp_output);
			// output_dir = "./";
		}

		// find which parameters are requested in the timevar file
		// these are the default spooky quantities
		setting = config_lookup(&config, "output.timevar_vars");
		spookyOutVar.length = config_setting_length( setting );
		std::printf("length timevar array = %d \n", spookyOutVar.length);
		// Allocate spooky output_vars
		spookyOutVar.name.resize(spookyOutVar.length);
		std::cout << "The following quantities will be computed: \t";
		for(i = 0 ; i < spookyOutVar.length ; i++) {
			temp_string = config_setting_get_string_elem( setting, i);
			std::cout << std::string(temp_string) << "\t";
			spookyOutVar.name[i] = std::string(temp_string);
		}
		std::cout << std::endl;


		// now for the user-defined quantities

		// if (!config_lookup_string(&config, "output.user_timevar_vars",&temp_output)){
		// 	std::cout << "Warning: you did not provide any variable in user outputs!" << std::endl;
		// }
		// else {
		// 	setting = config_lookup(&config, "output.user_timevar_vars");
		// 	userOutVar.length = config_setting_length( setting );
		// 	std::printf("length user timevar array = %d \n", userOutVar.length);
		// 	// Allocate user output_vars
		// 	userOutVar.name.resize(userOutVar.length);
		// 	std::cout << "The following user quantities will be computed: \t";
		// 	for(i = 0 ; i < userOutVar.length ; i++) {
		// 		temp_string = config_setting_get_string_elem( setting, i);
		// 		std::cout << std::string(temp_string) << "\t";
		// 		userOutVar.name[i] = std::string(temp_string);
		// 	}
		// 	std::cout << std::endl;
		// }

		setting = config_lookup(&config, "output.user_timevar_vars");
		if(setting == NULL) {
			std::cout << "Warning: you did not provide any variable in user outputs!" << std::endl;
			userOutVar.length = 0;
		}
		else {
			std::cout << "User outputs were provided" << std::endl;
			userOutVar.length = config_setting_length( setting );
			std::printf("length of the user timevar array = %d \n", userOutVar.length);
			// Allocate user output_vars
			userOutVar.name.resize(userOutVar.length);
			std::cout << "The following user quantities will be computed: \t";
			for(i = 0 ; i < userOutVar.length ; i++) {
				temp_string = config_setting_get_string_elem( setting, i);
				std::cout << std::string(temp_string) << "\t";
				userOutVar.name[i] = std::string(temp_string);
			}
			std::cout << std::endl;
		}

		if(!config_lookup_int(&config, "output.profile_dir",&tmp_v)) {
			profile_dir = 0;
		}
		else {
			profile_dir = (int) tmp_v;
		}

		// find which parameters are requested in the profile file
		// setting = config_lookup(&config, "output.profile_vars");

		// if(setting == NULL) {
		// 	// ERROR_HANDLER(ERROR_WARNING, "You did not provide any variable in profile outputs");
		// }
		// else {
  //
		// 	if (!profile_dir) {
		// 		ERROR_HANDLER(ERROR_WARNING, "profile_dir is set to zero");
		// 	}
  //
		// 	profile_vars.length = config_setting_length( setting );
  //
		// 	// Allocate output_vars
		// 	profile_vars.name = malloc( profile_vars.length * sizeof(char*) );
  //
		// 	for(i = 0 ; i < profile_vars.length ; i++) {
		// 		temp_string = config_setting_get_string_elem( setting, i);
  //
		// 		// Allocate the string
		// 		profile_vars.name[i] = malloc( sizeof(char) * (strlen(temp_string) + 1));
  //
		// 		// Copy the string in the right location
		// 		strcpy(profile_vars.name[i], temp_string);
		// 	}
		// }


		// Initial conditions parameters-------------------------------------------------------------------------
		if(!config_lookup_bool(&config, "init.vortex.enable",&init_vortex)) {
			init_vortex = 0;
		}
		if(!config_lookup_float(&config, "init.vortex.a",&vortex_a)) {
			vortex_a = 1.0;
		}
		if(!config_lookup_float(&config, "init.vortex.b",&vortex_b)) {
			vortex_b = 2.0;
		}
		if(!config_lookup_bool(&config, "init.spatial_structure",&init_spatial_structure)) {
			init_spatial_structure = 0;
		}
		if(!config_lookup_bool(&config, "init.large_scale_noise.enable",&init_large_scale_noise)) {
			init_large_scale_noise = 0;
		}
		if(!config_lookup_float(&config, "init.large_scale_noise.amplitude",&per_amplitude_large)) {
			per_amplitude_large = 0.0;
		}
		if(!config_lookup_float(&config, "init.large_scale_noise.cut_length",&noise_cut_length)) {
			noise_cut_length = 0.0;
		}
		if(!config_lookup_bool(&config, "init.large_scale_2D_noise.enable",&init_large_scale_2D_noise)) {
			init_large_scale_2D_noise = 0;
		}
		if(!config_lookup_float(&config, "init.large_scale_2D_noise.amplitude",&per_amplitude_large_2D)) {
			per_amplitude_large_2D = 0.0;
		}
		if(!config_lookup_float(&config, "init.large_scale_2D_noise.cut_length",&noise_cut_length_2D)) {
			noise_cut_length_2D = 0.0;
		}
		if(!config_lookup_bool(&config, "init.white_noise.enable",&init_white_noise)) {
			init_white_noise = 0;
		}
		// if(!config_lookup_bool(&config, "init.white_noise_vel.enable",&init_white_noise_vel)) {
		// 	init_white_noise_vel = 0;
		// }
		if(!config_lookup_float(&config, "init.white_noise.amplitude",&per_amplitude_noise)) {
			per_amplitude_noise = 0.0;
		}
		if(!config_lookup_bool(&config, "init.mean_field.enable",&init_mean_field)) {
			init_mean_field = 0;
		}
		if(!config_lookup_float(&config, "init.mean_field.bx0",&bx0)) {
			bx0 = 0.0;
		}
		if(!config_lookup_float(&config, "init.mean_field.by0",&by0)) {
			by0 = 0.0;
		}
		if(!config_lookup_float(&config, "init.mean_field.bz0",&bz0)) {
			bz0 = 0.0;
		}
		if(!config_lookup_bool(&config, "init.dump",&init_dump)) {
			init_dump = 0;
		}
		if(!config_lookup_bool(&config, "init.bench",&init_bench)) {
			init_bench = 0;
		}
		config_destroy(&config);
	// }
// #ifdef MPI_SUPPORT
// 	MPI_Bcast( &param, sizeof(struct Parameters), MPI_CHAR, 0, MPI_COMM_WORLD);
//
// 	// Copy varname structures properly (Broadcast does not work because of the allocation structure we use)
// 	if(rank !=0 ) {
// 		// Allocate the name list
// 		timevar_vars.name = malloc( timevar_vars.length * sizeof(char*) );
// 		profile_vars.name = malloc( profile_vars.length * sizeof(char*) );
// 	}
//
// 	// Next, allocate each name and copy it
// 	for(i = 0 ; i < timevar_vars.length ; i++) {
// 		if(rank==0) n = strlen(timevar_vars.name[i]);
//
// 		// Broadcast the string length and allocate it
// 		MPI_Bcast( &n, 1, MPI_INT, 0, MPI_COMM_WORLD);
//
// 		if(rank != 0) timevar_vars.name[i] = malloc( sizeof(char) * (n + 1));
//
// 		// Broadcast the string itself
// 		MPI_Bcast( timevar_vars.name[i], n+1, MPI_CHAR, 0, MPI_COMM_WORLD);
//
// 	}
//
// 	for(i = 0 ; i < profile_vars.length ; i++) {
// 		if(rank==0) n = strlen(profile_vars.name[i]);
//
// 		// Broadcast the string length and allocate it
// 		MPI_Bcast( &n, 1, MPI_INT, 0, MPI_COMM_WORLD);
//
// 		if(rank != 0) profile_vars.name[i] = malloc( sizeof(char) * (n + 1));
//
// 		// Broadcast the string itself
// 		MPI_Bcast( profile_vars.name[i], n+1, MPI_CHAR, 0, MPI_COMM_WORLD);
//
// 	}
//
// #endif

	// DEBUG_END_FUNC;

	return;
}
