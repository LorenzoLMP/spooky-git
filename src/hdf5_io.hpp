

#ifdef HEAT_EQ
    const std::string list_of_variables[NUM_FIELDS] = {"th"};
#endif

#ifdef INCOMPRESSIBLE
    #ifdef MHD
        #ifdef BOUSSINESQ
            const std::string list_of_variables[NUM_FIELDS] = {"vx","vy","vz","bx","by","bz","th"};
        #else
            const std::string list_of_variables[NUM_FIELDS] = {"vx","vy","vz","bx","by","bz"};
        #endif
    #else //not MHD
        #ifdef BOUSSINESQ
            const std::string list_of_variables[NUM_FIELDS] = {"vx","vy","vz","th"};
        #else //not BOUSSINESQ
            const std::string list_of_variables[NUM_FIELDS] = {"vx","vy","vz"};
        #endif
    #endif
#endif
