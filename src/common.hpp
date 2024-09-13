#include <array>
#include <complex>
#include <forward_list>
#include <vector>
// #include "spooky.hpp"
#include "define_types.hpp"
// #include "parameters.hpp"
// extern Parameters param;
#include "physics_modules.hpp"

#define KX 0
#define KY 1
#define KZ 2
// #ifdef SHEAR
//     #define KXt 3
// #endif

#define ASS 0
#define ADD 1

#ifdef INCOMPRESSIBLE
    #define VX 0
    #define VY 1
    #define VZ 2

    #ifdef MHD
        #define BX 3
        #define BY 4
        #define BZ 5
        #ifdef BOUSSINESQ
            #define TH 6
            #define NUM_FIELDS 7 // velocity plus mag plus temp
        #else
            #define NUM_FIELDS 6 // velocity plus mag field
        #endif
    #else //not MHD
        #ifdef BOUSSINESQ
            #define TH 3
            #define NUM_FIELDS 4 // velocity plus temp
        #else
            #define NUM_FIELDS 3 // only velocity
        #endif
    #endif
#endif

#ifdef HEAT_EQ
    #define TH 0
    #define NUM_FIELDS 1
#endif




// extern Parameters param;

