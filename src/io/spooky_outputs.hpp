#ifndef SPOOKY_OUTPUTS_HPP
#define SPOOKY_OUTPUTS_HPP

#include "common.hpp"
#include <vector>
#include <array>
#include <complex>

class Fields;
class Supervisor;

class SpookyOutput { // The class
    public:
        Supervisor *supervisor_ptr;
        int length; // Number of spooky variables
        // void* name; // Names of variables (need to be allocated properly)
        std::vector<std::string> name;
        Fields *field;
        // SpookyOutput(Fields *field_in);
        SpookyOutput(Supervisor &sup_in);
        ~SpookyOutput();
        scalar_type computeEnergy(data_type *vcomplex);
        scalar_type twoFieldCorrelation( scalar_type *v1, scalar_type *v2);
        scalar_type computeEnstrophy(data_type *vx,
                                    data_type *vy,
                                    data_type *vz);
        scalar_type computeDissipation(data_type *scalar_complex);
        scalar_type computeAnisoInjection(data_type* complex_Fields,
                                          scalar_type* real_Buffer);
        scalar_type computeAnisoDissipation(data_type* complex_Fields,
                                            scalar_type* real_Buffer);

};

#endif
