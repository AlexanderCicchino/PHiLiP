#ifndef __INVISCID_VORTEX_SHOCK_INTERACTION_H__
#define __INVISCID_VORTEX_SHOCK_INTERACTION_H__


#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/fe/mapping_q.h>
#include "tests.h"


#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "physics/physics_factory.h"
#include "physics/physics.h"
#include "ode_solver/ode_solver_factory.h"

#include<fenv.h>

namespace PHiLiP {
namespace Tests {

/// Inviscid Vortex Shock Interaction
template <int dim, int nstate>
class InviscidVortexShockInteraction : public TestsBase
{
public:
    /// Constructor.
    /** Simply calls the TestsBase constructor to set its parameters = parameters_input
     * */
    InviscidVortexShockInteraction(const Parameters::AllParameters *const parameters_input);

    /// Ensure that the kinetic energy is bounded.
    /** If the kinetic energy increases about its initial value, then the test should fail.
     *  Ref: Gassner 2016.
     * */
    int run_test() const override;

private:
    /// Computes the timestep from max eignevector.
    double get_timestep(const std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree, const double delta_x) const;
};


} //Tests
} //PHiLiP

#endif

