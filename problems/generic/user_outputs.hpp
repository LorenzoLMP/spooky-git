// #include "common.hpp"
#include "define_types.hpp"
#include <vector>
#include <array>
#include <complex>
#include "spooky_outputs.hpp"

class UserOutput : public SpookyOutput { // The class
    public:
        // int length; // Number of spooky variables
        // std::vector<std::string> name;
        UserOutput();
        ~UserOutput();
        scalar_type customFunction( data_type *vcomplex );
};

