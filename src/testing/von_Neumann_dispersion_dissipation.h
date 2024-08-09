#ifndef __VON_NEUMANN_DISPERSION_DISSIPATION_H__
#define __VON_NEUMANN_DISPERSION_DISSIPATION_H__

#include "tests.h"
#include "dg/dg_base.hpp"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Burgers' periodic unsteady test
template <int dim, int nstate>
class VonNeumannDispersionDissipation: public TestsBase
{
public:
    /// Constructor
    VonNeumannDispersionDissipation(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~VonNeumannDispersionDissipation() {};

    /// Run test
    int run_test () const override;

private:
    /// Computes avgerage solution value.
    std::array<std::vector<double>,nstate> compute_u_avg(const std::shared_ptr < DGBase<dim, double> > &dg, 
                                            const unsigned int poly_degree,
                                            const unsigned int n_elem,
                                            const double left, const double right) const;
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif
