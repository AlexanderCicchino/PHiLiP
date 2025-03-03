#ifndef __MIN_ENTROPY_PRINCIPLE_LIMITER__
#define __MIN_ENTROPY_PRINCIPLE_LIMITER__

#include "bound_preserving_limiter.h"
#include "physics/burgers.h"

namespace PHiLiP {
/// Class for implementation of min entropy principle limiter.
template<int dim, int nstate, typename real>
class MinEntropyPrincipleLimiter : public BoundPreservingLimiterState <dim, nstate, real>
{
public:
    /// Constructor
    explicit MinEntropyPrincipleLimiter(
        const Parameters::AllParameters* const parameters_input);

    /// Destructor
    ~MinEntropyPrincipleLimiter() = default;

    /// Pointer to TVB limiter class (TVB limiter can be applied in conjunction with this limiter)
    std::shared_ptr<BoundPreservingLimiterState<dim, nstate, real>> tvbLimiter;

    /// Pointer to positivity preserving limiter class
    std::shared_ptr<PositivityPreservingLimiter<dim, nstate, real>> posdensity_limiter;

    /// Euler physics pointer. Used to compute pressure.
    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_physics;
    
    /// Burgers physics pointer. Used to compute pressure.
    std::shared_ptr < Physics::Burgers<1, 1, double > > burgers_physics;

    /// PDE type.
    Parameters::AllParameters::PartialDifferentialEquation pde_type;

    /// Function to obtain the solution cell average
    using BoundPreservingLimiterState<dim, nstate, real>::get_soln_cell_avg;

    /// Min entropy in each cell.
    /* size number of local cells.
    */
    dealii::LinearAlgebra::distributed::Vector<double> local_min_entropy;

    /// Applies positivity-preserving limiter to the solution.
    /// Using Zhang,Shu November 2010 Eq 3.14-3.19 or Wang, Shu 2012 Eq 3.7
    /// we apply a limiter on the global solution
    void limit(
        dealii::LinearAlgebra::distributed::Vector<double>&     solution,
        const dealii::DoFHandler<dim>&                          dof_handler,
        const dealii::hp::FECollection<dim>&                    fe_collection,
        const dealii::hp::QCollection<dim>&                     volume_quadrature_collection,
        const unsigned int                                      grid_degree,
        const unsigned int                                      max_degree,
        const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
        const dealii::hp::QCollection<1>                        oneD_quadrature_collection);
protected:
    /// Obtain the theta value used to scale all the states using 3.16-3.18 in Zhang, Shu 2010
    std::vector<real> get_theta2_Zhang2010(
        const std::vector< real >&                      p_lim,
        const std::array<real, nstate>&                 soln_cell_avg,
        const std::array<std::vector<real>, nstate>&    soln_at_q,
        const unsigned int                              n_quad_pts,
        const double                                    eps,
        const double                                    gamma);

    /// Obtain the theta value used to scale all the states using 3.7 in Wang, Shu 2012
    real get_theta2_Wang2012(
        const std::array<std::vector<real>, nstate>&    soln_at_q,
        const unsigned int                              n_quad_pts,
        const double                                    p_avg);

    /// Obtain the value used to scale density and enforce positivity of density
    /// Using 3.15 from Zhang, Shu 2010
    real get_density_scaling_value(
        const double    density_avg,
        const double    density_min,
        const double    pos_eps,
        const double    p_avg);

    /// Function to verify the limited solution preserves positivity of density and pressure
    /// and write back limited solution
    void write_limited_solution(
        dealii::LinearAlgebra::distributed::Vector<double>&     solution,
        const std::array<std::vector<real>, nstate>&            soln_coeff,
        const unsigned int                                      n_shape_fns,
        const std::vector<dealii::types::global_dof_index>&     current_dofs_indices);

    /// Set's the minimum entropy value in each cell.
    void set_cell_min_entropy(
        dealii::LinearAlgebra::distributed::Vector<double>&     solution,
        const dealii::DoFHandler<dim>&                          dof_handler,
        const dealii::hp::FECollection<dim>&                    fe_collection,
        const dealii::hp::QCollection<dim>&                     volume_quadrature_collection,
        const unsigned int                                      grid_degree,
        const unsigned int                                      max_degree,
        const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
        const dealii::hp::QCollection<1>                        oneD_quadrature_collection);

}; // End of PositivityPreservingLimiter Class
} // PHiLiP namespace

#endif

