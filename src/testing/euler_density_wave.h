#ifndef __EULER_DENSITY_WAVE_H__
#define __EULER_DENSITY_WAVE_H__


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

/// Euler Taylor Green Vortex
/** Ensure that the kinetic energy is bounded.
 *  Ref: Gassner 2016.
 */
template <int dim, int nstate>
class EulerDensityWave : public TestsBase
{
public:
    /// Constructor.
    /** Simply calls the TestsBase constructor to set its parameters = parameters_input
     * */
    EulerDensityWave(const Parameters::AllParameters *const parameters_input);

    /// Ensure that the kinetic energy is bounded.
    /** If the kinetic energy increases about its initial value, then the test should fail.
     *  Ref: Gassner 2016.
     * */
    int run_test() const override;

private:
    /// Computes an integral of the kinetic energy (density * velocity squared) in the entire domain.
    /** Overintegration of kinetic energy.
     * */
    double compute_kinetic_energy(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const;
    /// Computes the conservtion.
    double compute_conservation(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const;
    /// Computes the entropy.
    double compute_entropy(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const;
    /// Computes the timestep from max eignevector.
    double get_timestep(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree, const double delta_x) const;
};


} //Tests
} //PHiLiP

#endif



