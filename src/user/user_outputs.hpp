#include "common.hpp"
#include <vector>
#include <array>
#include <complex>
#include "spooky_outputs.hpp"

// class SpookyOutput;
class Supervisor;

class UserOutput : public SpookyOutput { // The class
    public:
        // int length; // Number of spooky variables
        // std::vector<std::string> name;
        UserOutput(Supervisor &sup_in);
        // ~UserOutput();
        scalar_type customFunction( data_type *vcomplex );
        scalar_type computekpartheta(data_type* complex_Fields,
                                    scalar_type* real_Buffer);
};

