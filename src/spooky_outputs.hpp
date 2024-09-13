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
        scalar_type computeEnergy(data_type *d_all_tmparray);
        SpookyOutput();
        ~SpookyOutput();
};

