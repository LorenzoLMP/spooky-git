#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
// #include "libconfig/libconfig.h++"
#include <libconfig.h>
#include <math.h>
#include "common.hpp"
#include "parameters.hpp"
// #include "spooky_outputs.hpp"

#define SPOOKY_CONFIG_FILENAME		"spooky.cfg"


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
Parameters::Parameters(Supervisor& sup_in, std::string input_dir) : spookyOutVar(sup_in), userOutVar(sup_in) {
// void Parameters::read_Parameters() {
	// field = fields_in;
	supervisor_ptr = &sup_in;
	// Read the config file and initialize everyting
	config_t	config;		// Initialize the structure
	config_setting_t * setting;	// a setting structure
	int tmp_v;
	int i;
	std::string config_fullpath(input_dir);
	config_fullpath.append("/");
	config_fullpath.append(std::string(SPOOKY_CONFIG_FILENAME));

	const char * CONFIG_FILENAME = config_fullpath.c_str();

	const char * configname;

	const char * temp_string;

    const char * temp_output;


	config_init(&config);

	if(!config_read_file(&config, CONFIG_FILENAME)) {
		std::printf("Error reading configuration file in line %d: %s\n", config_error_line(&config), config_error_text(&config));
		exit(0);
		// ERROR_HANDLER(ERROR_CRITICAL, "Failed to read the configuration file");
	}

	if(config_lookup_string(&config,"configname",&configname)) {
		std::printf("Using config file: %s at %s.\n",configname,CONFIG_FILENAME);
	}

	/*******************
	 *
	 * modules section
	 *
	 ******************/

	if(!config_lookup_int(&config, "modules.debug",&debug)) {
		debug = 0;
	}
	if(!config_lookup_bool(&config, "modules.incompressible",&incompressible)) {
		incompressible = 0;
	}
	if(!config_lookup_bool(&config, "modules.heat_equation",&heat_equation)) {
		heat_equation = 0;
	}
	// if(!config_lookup_bool(&config, "modules.explicit_dissipation",&explicit_dissipation)) {
	// 	explicit_dissipation = 0;
	// }
	if(!config_lookup_bool(&config, "modules.boussinesq",&boussinesq)) {
		boussinesq = 0;
	}
	if(!config_lookup_bool(&config, "modules.stratification",&stratification)) {
		stratification = 0;
	}
	if(!config_lookup_int(&config, "modules.strat_direction",&strat_direction)) {
		strat_direction = 2; // by default in the z direction
	}
	if(!config_lookup_bool(&config, "modules.mhd",&mhd)) {
		mhd = 0;
	}
	if(!config_lookup_bool(&config, "modules.anisotropic_diffusion",&anisotropic_diffusion)) {
		anisotropic_diffusion = 0;
	}
	if(!config_lookup_bool(&config, "modules.supertimestepping",&supertimestepping)) {
		// supertimestepping = 1;
		supertimestepping = 0;
		// std::printf("With sts. \n");
	}
	std::printf("supertimestepping = %d \n",supertimestepping);
	if (supertimestepping) {
		int sts_length;
		if(config_lookup_string(&config, "modules.sts_algorithm",&temp_string)) {
			std::printf("sts_algorithm: %s \n",temp_string);
			sts_algorithm = std::string(temp_string);

			setting = config_lookup(&config, "modules.sts_variables");


			if(setting == NULL) {
				std::cout << "Warning: you did not provide any variable for sts! Therefore all parabolic terms will be evolved as normal." << std::endl;
				sts_variables.resize(0);
				supertimestepping = 0;
			}
			else {
				std::cout << "sts variables were provided" << std::endl;
				sts_length = config_setting_length( setting );

				if (sts_length > 0) {
					std::printf("Number of sts variables = %d \n", sts_length);
					sts_variables.resize(sts_length);
					std::cout << "The following sts variables quantities will be evolved: \t";
					for(i = 0 ; i < sts_length ; i++) {
						temp_string = config_setting_get_string_elem( setting, i);
						std::cout << std::string(temp_string) << "\t";
						sts_variables[i] = std::string(temp_string);
					}
					std::cout << std::endl;
				}
				else {
					std::cout << "Warning: you did not provide any variable for sts! Therefore all parabolic terms will be evolved as normal." << std::endl;
					sts_variables.resize(0);
					supertimestepping = 0;
				}
			}
		}
	}
	else {
		sts_variables.resize(0);
	}
	if(!config_lookup_bool(&config, "modules.shearing",&shearing)) {
		shearing = 0;
	}
	if(!config_lookup_bool(&config, "modules.rotating",&rotating)) {
		rotating = 0;
	}


	/*******************
	 *
	 * physics section
	 *
	 ******************/

	if(!config_lookup_float(&config, "physics.boxsize.[0]",&lx)) {
		lx = 1.0;
	}
	if(!config_lookup_float(&config, "physics.boxsize.[1]",&ly)) {
		ly = 1.0;
	}
	if(!config_lookup_float(&config, "physics.boxsize.[2]",&lz)) {
		lz = 1.0;
	}

	if(!config_lookup_int(&config, "physics.gridsize.[0]",&nx)) {
		nx = 32;
	}
	if(!config_lookup_int(&config, "physics.gridsize.[1]",&ny)) {
		ny = 32;
	}
	if(!config_lookup_int(&config, "physics.gridsize.[2]",&nz)) {
		nz = 32;
	}

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
	if(!config_lookup_float(&config, "physics.omega",&omega)) {
		omega = 0.0;
	}
// #ifndef WITH_ROTATION
// 		// Omega should be forced to zero in order to be fool-proof
// 		omega = 0.0;
// #endif
	if(!config_lookup_float(&config, "physics.shear",&shear)) {
		shear = 0.0;
	}
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

	/*******************
	 *
	 * Code parameters
	 *
	 ******************/

	if(!config_lookup_float(&config, "code.cfl",&cfl)) {
		cfl = 1.5;
	}
	if(!config_lookup_float(&config, "code.cfl_par",&cfl_par)) {
		cfl_par = 1.0;
	}
	if(!config_lookup_int(&config, "code.safety_sts",&safety_sts)) {
		safety_sts = 1;
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
	if(!config_lookup_float(&config, "code.max_walltime_elapsed",&max_walltime_elapsed)) {
		max_walltime_elapsed = 42;
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

	/*******************
	 *
	 * Output parameters
	 *
	 ******************/

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
	if(setting == NULL) {
		std::cout << "Warning: you did not provide any variable to compute!" << std::endl;
		spookyOutVar.length_timevar = 0;
	}
	else {
		spookyOutVar.length_timevar = config_setting_length( setting );
		std::printf("length timevar array = %d \n", spookyOutVar.length_timevar);
		// Allocate spooky output_vars
		spookyOutVar.name_timevar.resize(spookyOutVar.length_timevar);
		std::cout << "The following quantities will be computed: \t";
		for(i = 0 ; i < spookyOutVar.length_timevar ; i++) {
			temp_string = config_setting_get_string_elem( setting, i);
			std::cout << std::string(temp_string) << "\t";
			spookyOutVar.name_timevar[i] = std::string(temp_string);
		}
		std::cout << std::endl;
	}

	
	// find which parameters are requested in the spectra file
	setting = config_lookup(&config, "output.spectra_vars");
	if(setting == NULL) {
		std::cout << "Warning: you did not provide any spectra to compute!" << std::endl;
		spookyOutVar.length_spectra = 0;
	}
	else { 
		spookyOutVar.length_spectra = config_setting_length( setting );
		std::printf("length spectra array = %d \n", spookyOutVar.length_spectra);
		// Allocate spooky output_vars
		spookyOutVar.name_spectra.resize(spookyOutVar.length_spectra);
		std::cout << "The following quantities will be computed: \t";
		for(i = 0 ; i < spookyOutVar.length_spectra ; i++) {
			temp_string = config_setting_get_string_elem( setting, i);
			std::cout << std::string(temp_string) << "\t";
			spookyOutVar.name_spectra[i] = std::string(temp_string);
		}
		std::cout << std::endl;
	}
	

	// now for the user-defined quantities
	setting = config_lookup(&config, "output.user_timevar_vars");
	if(setting == NULL) {
		std::cout << "Warning: you did not provide any variable in user outputs!" << std::endl;
		userOutVar.length_timevar = 0;
	}
	else {
		std::cout << "User outputs were provided" << std::endl;
		userOutVar.length_timevar = config_setting_length( setting );
		std::printf("length of the user timevar array = %d \n", userOutVar.length_timevar);
		// Allocate user output_vars
		userOutVar.name_timevar.resize(userOutVar.length_timevar);
		std::cout << "The following user quantities will be computed: \t";
		for(i = 0 ; i < userOutVar.length_timevar ; i++) {
			temp_string = config_setting_get_string_elem( setting, i);
			std::cout << std::string(temp_string) << "\t";
			userOutVar.name_timevar[i] = std::string(temp_string);
		}
		std::cout << std::endl;
	}
	if(!config_lookup_int(&config, "output.profile_dir",&tmp_v)) {
		profile_dir = 0;
	}
	else {
		profile_dir = (int) tmp_v;
	}

	/*******************
	 *
	 * initial conditions
	 *
	 ******************/

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

	return;
}

int Parameters::checkParameters(){

	int paramsConsistent = 1;

	if (incompressible and heat_equation) {
		paramsConsistent = 0;
		std::cout << "Error: incompressible and heat_equation modules are mutually exclusive" << std::endl;
	}
	if (not incompressible and not heat_equation) {
		paramsConsistent = 0;
		std::cout << "Error: you have to chose either incompressible or heat_equation module" << std::endl;
	}
	if (boussinesq and not incompressible){
		paramsConsistent = 0;
		std::cout << "Error: Boussinesq requires incompressible module" << std::endl;
	}
	if (stratification and not boussinesq){
		paramsConsistent = 0;
		std::cout << "Error: stratification requires Boussinesq module" << std::endl;
	}
	if (mhd and not incompressible){
		paramsConsistent = 0;
		std::cout << "Error: mhd requires incompressible module" << std::endl;
	}
	if (anisotropic_diffusion and not (mhd and boussinesq) ) {
		paramsConsistent = 0;
		std::cout << "Error: anisotropic_diffusion requires mhd and boussinesq module" << std::endl;
	}
	if (supertimestepping) {
		if (sts_variables.size() > vars.NUM_FIELDS) {
			paramsConsistent = 0;
			std::cout << "Error: the number of sts variables is larger than the total number of variables." << std::endl;
		}
	}

	// is shearing is on, overwrite toutput_flow so that
	// it will output only when the domain is exactly periodic
	if (shearing) {
		double tperiodic = ly / (shear * lx);
		if (toutput_flow < tperiodic){
			toutput_flow = tperiodic;
		}
		else {
			toutput_flow = (double) int(toutput_flow/tperiodic);
		}
		std::cout << "Overwriting output for data snapshots because of shearing" << std::endl;
		std::printf("toutput_flow = %.4e \n",toutput_flow);
	}

	return paramsConsistent;
}

void Parameters::popVariablesGrid() {

	vars.NUM_FIELDS = 0;
	vars.VX = 0; vars.VY = 0; vars.VZ = 0;
    vars.BX = 0; vars.BY = 0; vars.BZ = 0;
    vars.TH = 0;
    vars.VEL = 0; vars.MAG = 0;


    // int NUM_FIELDS;

	vars.KX = 0;
	vars.KY = 1;
	vars.KZ = 2;

	if (heat_equation) {
		vars.TH = 0;
		vars.NUM_FIELDS += 1;
	}
	else if (incompressible) {
		vars.VX = 0;
		vars.VY = 1;
		vars.VZ = 2;
		vars.NUM_FIELDS += 3;
		vars.VEL = 0;

		if (mhd) {
			vars.BX = 3;
			vars.BY = 4;
			vars.BZ = 5;
			vars.NUM_FIELDS += 3;
			vars.MAG = 3;

			if (boussinesq) {
				vars.TH = 6;
				vars.NUM_FIELDS += 1;
			}
		}
		else { // not mhd
			if (boussinesq) {
				vars.TH = 3;
				vars.NUM_FIELDS += 1;
			}
		}
	}

	// set name of variables (for output)
	vars.VAR_LIST.resize(vars.NUM_FIELDS);

	if (heat_equation) {
		vars.VAR_LIST = {"th"};
	}
	else if (incompressible) {

		if (mhd) {
			if (boussinesq) {
				vars.VAR_LIST = {"vx","vy","vz","bx","by","bz","th"};
			}
			else {
				vars.VAR_LIST = {"vx","vy","vz","bx","by","bz"};
			}
		}
		else { // not mhd
			if (boussinesq) {
				vars.VAR_LIST = {"vx","vy","vz","th"};
			}
			else {
				vars.VAR_LIST = {"vx","vy","vz"};
			}
		}
	}

	grid.NX =  nx;
	grid.NY =  ny;
	grid.NZ =  nz;

	grid.FFT_SIZE[0] = (size_t) grid.NX;
	grid.FFT_SIZE[1] = (size_t) grid.NY;
	grid.FFT_SIZE[2] = (size_t) grid.NZ;

	grid.NTOTAL = grid.NX * grid.NY * grid.NZ;

	grid.NTOTAL_COMPLEX = grid.NX * grid.NY * (( grid.NZ / 2) + 1);



}
