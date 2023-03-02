#ifndef __BURGERS_LINEAR_TABILITY_H__
#define __BURGERS_LINEAR_TABILITY_H__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Burgers' periodic unsteady test
template <int dim, int nstate>
class BurgersLinearStability: public TestsBase
{
public:
    /// Constructor
    BurgersLinearStability(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~BurgersLinearStability() {};

    /// Run test
    int run_test () const override;
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif

