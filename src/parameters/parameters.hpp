#include <iostream>
// #include "spooky_outputs.hpp"
#include <vector>
// #include <array>
// #include <complex>
#include "common.hpp"
#include "user_outputs.hpp"
// #include "fields.hpp"


// use forward declarations in the header files to get around the circular dependencies
// https://stackoverflow.com/questions/994253/two-classes-that-refer-to-each-other
// class Fields;
class Supervisor;

// class SpookyOutput;

// class UserOutput;

class Parameters {
  public:

	Supervisor *supervisor_ptr;

	/*******************
	 *
	 * modules section
	 *
	 ******************/

	int debug;   				// debug levels = 0 (no debug), 1 (less verbose), 2 (very verbose)
	int incompressible;			// HD module
	int heat_equation;			// heat_equation module (incompatible with incompressible)
	// int explicit_dissipation; 	// need to check this
	int boussinesq; 			// add boussinesq module (only with incompressible)
	int stratification;			// when boussinesq = true, if background stratification
	int strat_direction;		// with boussinesq and stratification, direction of strat (0=x, 1=y, 2=z)
	int mhd;					// magnetic fields
	int anisotropic_diffusion; 	// anisotropic diffusion of heat along magnetic fields
	int supertimestepping; 		// supertimestepping of parabolic terms
	std::string sts_algorithm; 	// supertimestepping algorithm. Valid choices: "sts", "rkl2"
	std::vector<std::string> sts_variables;
	int shearing; 			// add background shearing
	int rotating;			// add background rotation

	/*******************
	 *
	 * physics section
	 *
	 ******************/

    double lx;				/**< Box length in X*/
	double ly;				/**< Box length in Y*/
	double lz;				/**< Box length in Z*/

	int nx;			// real grid size in x-direction
	int ny;			// real grid size in y-direction
	int nz;			// real grid size in z-direction

	double reynolds;		/**< Reynolds number (actully the inverse of the viscosity) */
	double nu;
	double reynolds_m;		/**< Magnetic Reynolds number (actully the inverse of the resistivity)  Used only when MHD is on*/
	double nu_m;
	double reynolds_th;		/**< Thermal Reynolds number (actully the inverse of the thermal diffusivity)  Used only when Boussinesq is on*/
	double nu_th;
	double reynolds_B;		/**< Reynolds number based on Braginskii viscosity */
	double reynolds_ani;		/**< Anisotropic Thermal Reynolds number (actully the inverse of the anisotropic thermal diffusivity)  Used only when Boussinesq is on*/
	double x_hall;			/**< Hall parameter */
	double N2;				/**< Brunt Vaissala frequency squared */
	double OmegaT2;				/**< MTI frequency squared */
	double ALPHA;				/**< spatially dependent thermal conduction coefficient*/
	double omega;			/**< Vertical rotation rate (if Shear=1, Keplerian if found for (2.0/3.0). Only when WITH_ROTATION is on. */
	double shear;			/**< Shear rate (only when WITH_SHEAR is on) */
	double omega_shear;		/**< Pulsation of the time dependant shear (only when WITH_SHEAR and TIME_DEPENDANT_SHEAR is on, or alternatively WITH_LINEAR_TIDE) */
	double cs;				/**< Sound speed (only used when compressible is on) */

	// Particles parameters
	int    particles_n;		/**< Number of particles */
	double particles_mass;  /**< Mass of the particles */
	double particles_stime;	/**< Stopping time of the particles */
	double particles_dg_ratio;	/**< Dust to gas mass ratio for the particles feedback */
	double particles_epsilon;	/**< Pressure gradient epsilon */


	/*******************
	 *
	 * Code parameters
	 *
	 ******************/


	double cfl;				/**< CFL safety factor. Should be smaller than sqrt(3) for RK3 to be stable.*/
	double cfl_par;				/**< CFL safety factor for parabolic STS*/
	int safety_sts;     //num max of sts steps allowed
	double safety_source;	/**< Safety factor for SHEAR, Coriolis and Boussinesq terms (should be ~0.2 for accuracy) */
	double t_initial;		/**< Initial time of the simulation */
	double t_final;			/**< Simulation will stop if it reaches this time */
	double max_walltime_elapsed;	/**< Maximum wall-clock elapsed time (in hours). Will stop after this elapsed time */
	int    interface_check;	/**< Number of loops between two checks for a user input. On slow filesystems, increase this number */
	int    interface_output_file;	/**< Set this option to force code outputs to a file instead of the screen */
	int    force_symmetries;	/**< set to enforce spectral symmetries and mean flow to zero. Useful when N^2 or kappa^2 < 0. (see enforce_symm() )*/
	int    symmetries_step;		/**< Number of loops between which the symmetries are enforced. Should be around ~20 for Boussinesq convection*/
	int    antialiasing;		/**< 2/3 Antialisaing rule. Could be removed if you assume is unimportant (untested feature). */
	int    restart;


	/*******************
	 *
	 * Output parameters
	 *
	 ******************/

	double toutput_time;		/**< Time between two outputs in the timevar file */
	// double toutput_spec;		/**< Time between two outputs in the spectrum files */
	double toutput_flow;		/**< Time between two snapshot outputs */
	double toutput_dump;		/**< Time between two restart dump outputs (restart dump are erased) */
    // char   output_dir[256];
    std::string  output_dir;
	int        profile_dir;                 /**< Direction of the profile. 0= no profile, 1= x direciton, 2= y direction, 3= z direction */
	int		output_vorticity;	/**< Output the vorticity field in the 3D snapshots */

	SpookyOutput spookyOutVar;	/**< Name of the variables needed in the timevar file */
	UserOutput userOutVar;	/**< User-defined output vars */


	/*******************
	 *
	 * initial conditions
	 *
	 ******************/

	int	   init_vortex;			/**< Add a 2D Kida vortex in the box. Assumes S=1. Requires b>a*/
	double vortex_a;			/**< x dimension of the vortex */
	double vortex_b;			/**< y dimension of the vortex */
	int    init_spatial_structure;	/**< Init a user-defined spatial structure (see initflow.c) */
	int	   init_large_scale_noise;	/**< Init a large scale random noise */
	double per_amplitude_large;		/**< Amplitude of the large scale random noise */
	double noise_cut_length;		/**< Wavelength over which the noise is applied */
	int	   init_large_scale_2D_noise;	/**< Init a large scale 2D (x,y) random noise  */
	double per_amplitude_large_2D;		/**< Amplitude of the 2D large scale random noise */
	double noise_cut_length_2D;		    /**< Wavelength over which the 2D noise is applied */
	int    init_white_noise;		/**< Init a random white noise on all the modes */
	double per_amplitude_noise;		/**< total amplitude of the perturbation */
	int    init_mean_field;			/**< Force the mean magnetic field to a given value. */
	double bx0;						/**< Mean magnetic field in the x direction */
	double by0;						/**< Mean magnetic field in the y direction */
	double bz0;						/**< Mean magnetic field in the z direction */

	int    init_dump;				/**< Use a dump file as an initial condition (everything else, including t, noutput (...) is reinitialised) */
	int	   init_bench;				/**< Init the Benchmark initial conditions */


	// Parameters(Fields *fields_in, std::string input_dir);
	Parameters(Supervisor& sup_in);
    ~Parameters();

	int checkParameters();
	void popVariablesGrid();
    // void read_Parameters(std::string input_dir);
	// void read_Parameters();
};

