#ifndef __NONSYMMETRIC_CURVED_PERIODIC_GRID_CHAN_H__
#define __NONSYMMETRIC_CURVED_PERIODIC_GRID_CHAN_H__

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/distributed/tria.h>

namespace PHiLiP {
namespace Grids {

/// Create a nonsymmetric grid with an associated nonlinear manifold.
/** The mapping for the 3D grids follows from
 *  Section 6.3.1 from Chan, Jesse, and Lucas C. Wilcox. "On discretely entropy stable weight-adjusted discontinuous Galerkin methods: curvilinear meshes." Journal of Computational Physics 378 (2019): 366-393.
*/
template<int dim, typename TriangulationType>
void nonsymmetric_curved_grid_chan(
    TriangulationType &grid,
    const unsigned int n_elements_per_dim);

/// Nonsymmetric manifold.
template<int dim,int spacedim,int chartdim>
class NonsymmetricCurvedGridManifoldChan : public dealii::ChartManifold<dim,spacedim,chartdim> {
protected:
    static constexpr double pi = atan(1) * 4.0; ///< PI.
    const double beta = 1.0/20.0;

public:
    /// Constructor.
    NonsymmetricCurvedGridManifoldChan()
    : dealii::ChartManifold<dim,spacedim,chartdim>() {};
    
    /// Templated mapping from square to the nonsymmetric warping.
    template<typename real>
    dealii::Point<spacedim,real> mapping(const dealii::Point<chartdim,real> &chart_point) const;

    virtual dealii::Point<chartdim> pull_back(const dealii::Point<spacedim> &space_point) const override; ///< See dealii::Manifold.
    virtual dealii::Point<spacedim> push_forward(const dealii::Point<chartdim> &chart_point) const override; ///< See dealii::Manifold.
    virtual dealii::DerivativeForm<1,chartdim,spacedim> push_forward_gradient(const dealii::Point<chartdim> &chart_point) const override; ///< See dealii::Manifold.
    virtual std::unique_ptr<dealii::Manifold<dim,spacedim> > clone() const override; ///< See dealii::Manifold.
};

} // namespace Grids
} // namespace PHiLiP
#endif

