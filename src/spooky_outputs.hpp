// #include "common.hpp"
#include "define_types.hpp"
#include <vector>
#include <array>
#include <complex>

class SpookyOutput { // The class
    public:
        int length; // Number of spooky variables
        // void* name; // Names of variables (need to be allocated properly)
        std::vector<std::string> name;
        SpookyOutput();
        ~SpookyOutput();
        scalar_type computeEnergy(data_type *vcomplex);
        scalar_type twoFieldCorrelation( scalar_type *v1, scalar_type *v2);
        scalar_type computeEnstrophy(data_type *v_all_complex,
                                        scalar_type *d_all_kvec,
                                        data_type *tmparray);
};

