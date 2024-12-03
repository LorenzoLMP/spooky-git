// #include "common.hpp"
#include "define_types.hpp"
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
        SpookyOutput();
        ~SpookyOutput();
        scalar_type computeEnergy(data_type *vcomplex);
        scalar_type twoFieldCorrelation( scalar_type *v1, scalar_type *v2);
        scalar_type computeEnstrophy(data_type *v_all_complex,
                                        scalar_type *d_all_kvec,
                                        data_type *tmparray);
        scalar_type computeDissipation(data_type *scalar_complex,
                                             scalar_type *d_all_kvec,
                                             data_type *tmparray);
        scalar_type computeAnisoInjection(scalar_type *d_all_kvec,
                                            data_type *d_all_fields,
                                            data_type **d_farray,
                                            scalar_type **d_farray_r,
                                            data_type *d_all_tmparray,
                                            data_type **d_tmparray,
                                            scalar_type **d_tmparray_r,
                                            scalar_type *d_mask,
                                                  int num_fields);
        scalar_type computeAnisoDissipation(scalar_type *d_all_kvec,
                                                  data_type *d_all_fields,
                                                  data_type **d_farray,
                                                  scalar_type **d_farray_r,
                                                  data_type *d_all_tmparray,
                                                  data_type **d_tmparray,
                                                  scalar_type **d_tmparray_r,
                                                  scalar_type *d_mask,
                                                  int num_fields);

};

