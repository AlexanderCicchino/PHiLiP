#ifndef __BURGERS_STABILITY_H__
#define __BURGERS_STABILITY_H__

#include "tests.h"
#include "dg/dg.h"
#include "physics/physics_factory.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Burgers' periodic unsteady test
template <int dim, int nstate>
class BurgersEnergyStability: public TestsBase
{
public:
    /// Constructor
    BurgersEnergyStability(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~BurgersEnergyStability() {};

    /// Run test
    int run_test () const override;
private:
    /// Function computes the energy
    double compute_energy(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg) const;
    /// Function computes the conservation
    double compute_conservation(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg, const double poly_degree) const;
    ///Computes change in entropy in the norm.
    /** That is let \f$ v\f$ represent the entropy variables, it computes \f$v(M+K)\frac{du}{dt}^T\f$. 
     */
   // double compute_change_in_entropy(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const;
    std::array<double,2> compute_change_in_entropy(const std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const;

    /// Computes the timestep from max eignevector.
    double get_timestep(const std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree, const double delta_x) const;
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif
