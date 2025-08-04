#include "ADTypes.hpp"

#include "convective_numerical_flux.hpp"

namespace PHiLiP {
namespace NumericalFlux {

using AllParam = Parameters::AllParameters;

// Protyping low level functions
template<int nstate, typename real_tensor>
std::array<real_tensor, nstate> array_average(
    const std::array<real_tensor, nstate> &array1,
    const std::array<real_tensor, nstate> &array2)
{
    std::array<real_tensor,nstate> array_average;
    for (int s=0; s<nstate; s++) {
        array_average[s] = 0.5*(array1[s] + array2[s]);
    }
    return array_average;
}

template <int dim, int nstate, typename real>
NumericalFluxConvective<dim, nstate, real>::NumericalFluxConvective(
    std::unique_ptr< BaselineNumericalFluxConvective<dim,nstate,real> > baseline_input,
    std::unique_ptr< RiemannSolverDissipation<dim,nstate,real> >   riemann_solver_dissipation_input)
    : baseline(std::move(baseline_input))
    , riemann_solver_dissipation(std::move(riemann_solver_dissipation_input))
{ }

template<int dim, int nstate, typename real>
std::array<real, nstate> NumericalFluxConvective<dim,nstate,real>
::evaluate_flux (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const
{
    // baseline flux (without upwind dissipation)
    const std::array<real, nstate> baseline_flux_dot_n 
        = this->baseline->evaluate_flux(soln_int, soln_ext, normal_int);

    // Riemann solver dissipation
    const std::array<real, nstate> riemann_solver_dissipation_dot_n 
        = this->riemann_solver_dissipation->evaluate_riemann_solver_dissipation(soln_int, soln_ext, normal_int);

    // convective numerical flux: sum of baseline and Riemann solver dissipation term
    std::array<real, nstate> numerical_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        numerical_flux_dot_n[s] = baseline_flux_dot_n[s] + riemann_solver_dissipation_dot_n[s];
    }
    return numerical_flux_dot_n;
}

template <int dim, int nstate, typename real>
LaxFriedrichs<dim, nstate, real>::LaxFriedrichs(
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : NumericalFluxConvective<dim,nstate,real>(
        std::make_unique< CentralBaselineNumericalFluxConvective<dim, nstate, real> > (physics_input), 
        std::make_unique< LaxFriedrichsRiemannSolverDissipation<dim, nstate, real> > (physics_input))
{}

template <int dim, int nstate, typename real>
RoePike<dim, nstate, real>::RoePike(
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : NumericalFluxConvective<dim,nstate,real>(
        std::make_unique< CentralBaselineNumericalFluxConvective<dim, nstate, real> > (physics_input), 
        std::make_unique< RoePikeRiemannSolverDissipation<dim, nstate, real> > (physics_input))
{}

template <int dim, int nstate, typename real>
L2Roe<dim, nstate, real>::L2Roe(
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : NumericalFluxConvective<dim,nstate,real>(
        std::make_unique< CentralBaselineNumericalFluxConvective<dim, nstate, real> > (physics_input), 
        std::make_unique< L2RoeRiemannSolverDissipation<dim, nstate, real> > (physics_input))
{}

template <int dim, int nstate, typename real>
Central<dim, nstate, real>::Central(
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : NumericalFluxConvective<dim,nstate,real>(
        std::make_unique< CentralBaselineNumericalFluxConvective<dim, nstate, real> > (physics_input), 
        std::make_unique< ZeroRiemannSolverDissipation<dim, nstate, real> > ())
{}

template <int dim, int nstate, typename real>
EntropyConserving<dim, nstate, real>::EntropyConserving(
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : NumericalFluxConvective<dim,nstate,real>(
        std::make_unique< EntropyConservingBaselineNumericalFluxConvective<dim, nstate, real> > (physics_input), 
        std::make_unique< ZeroRiemannSolverDissipation<dim, nstate, real> > ())
{}

template <int dim, int nstate, typename real>
EntropyConservingWithLaxFriedrichsDissipation<dim, nstate, real>::EntropyConservingWithLaxFriedrichsDissipation(
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : NumericalFluxConvective<dim,nstate,real>(
        std::make_unique< EntropyConservingBaselineNumericalFluxConvective<dim, nstate, real> > (physics_input), 
        std::make_unique< LaxFriedrichsRiemannSolverDissipation<dim, nstate, real> > (physics_input))
{}

template <int dim, int nstate, typename real>
EntropyConservingWithRoeDissipation<dim, nstate, real>::EntropyConservingWithRoeDissipation(
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : NumericalFluxConvective<dim,nstate,real>(
        std::make_unique< EntropyConservingBaselineNumericalFluxConvective<dim, nstate, real> > (physics_input), 
        std::make_unique< RoePikeRiemannSolverDissipation<dim, nstate, real> > (physics_input))
{}

template <int dim, int nstate, typename real>
EntropyConservingWithL2RoeDissipation<dim, nstate, real>::EntropyConservingWithL2RoeDissipation(
    std::shared_ptr<Physics::PhysicsBase<dim, nstate, real>> physics_input)
    : NumericalFluxConvective<dim,nstate,real>(
        std::make_unique< EntropyConservingBaselineNumericalFluxConvective<dim, nstate, real> > (physics_input), 
        std::make_unique< L2RoeRiemannSolverDissipation<dim, nstate, real> > (physics_input))
{}

template <int dim, int nstate, typename real>
std::array<real, nstate> CentralBaselineNumericalFluxConvective<dim,nstate,real>::evaluate_flux(
 const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const
{
    using RealArrayVector = std::array<dealii::Tensor<1,dim,real>,nstate>;
    RealArrayVector conv_phys_flux_int;
    RealArrayVector conv_phys_flux_ext;

    conv_phys_flux_int = pde_physics->convective_flux (soln_int);
    conv_phys_flux_ext = pde_physics->convective_flux (soln_ext);
    
    RealArrayVector flux_avg;
    for (int s=0; s<nstate; s++) {
        flux_avg[s] = 0.0;
        for (int d=0; d<dim; ++d) {
            flux_avg[s][d] = 0.5*(conv_phys_flux_int[s][d] + conv_phys_flux_ext[s][d]);
        }
    }

    std::array<real, nstate> numerical_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        real flux_dot_n = 0.0;
        for (int d=0; d<dim; ++d) {
            flux_dot_n += flux_avg[s][d]*normal_int[d];
        }
        numerical_flux_dot_n[s] = flux_dot_n;
    }
    return numerical_flux_dot_n;
}

template <int dim, int nstate, typename real>
std::array<real, nstate> EntropyConservingBaselineNumericalFluxConvective<dim,nstate,real>::evaluate_flux(
 const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const
{
    using RealArrayVector = std::array<dealii::Tensor<1,dim,real>,nstate>;
    RealArrayVector conv_phys_split_flux;

    conv_phys_split_flux = pde_physics->convective_numerical_split_flux (soln_int,soln_ext);

    // Scalar dissipation
    std::array<real, nstate> numerical_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        real flux_dot_n = 0.0;
        for (int d=0; d<dim; ++d) {
            flux_dot_n += conv_phys_split_flux[s][d] * normal_int[d];
        }
        numerical_flux_dot_n[s] = flux_dot_n;
    }
    return numerical_flux_dot_n;
}

template<int dim, int nstate, typename real>
std::array<real, nstate> ZeroRiemannSolverDissipation<dim,nstate,real>
::evaluate_riemann_solver_dissipation (
    const std::array<real, nstate> &/*soln_int*/,
    const std::array<real, nstate> &/*soln_ext*/,
    const dealii::Tensor<1,dim,real> &/*normal_int*/) const
{
    // zero upwind dissipation
    std::array<real, nstate> numerical_flux_dot_n;
    numerical_flux_dot_n.fill(0.0);
    return numerical_flux_dot_n;
}

template<int dim, int nstate, typename real>
std::array<real, nstate> LaxFriedrichsRiemannSolverDissipation<dim,nstate,real>
::evaluate_riemann_solver_dissipation (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const
{
    const real conv_max_eig_int = pde_physics->max_convective_normal_eigenvalue(soln_int,normal_int);
    const real conv_max_eig_ext = pde_physics->max_convective_normal_eigenvalue(soln_ext,normal_int);
    // Replaced the std::max with an if-statement for the AD to work properly.
    //const real conv_max_eig = std::max(conv_max_eig_int, conv_max_eig_ext);
    real conv_max_eig;
    if (conv_max_eig_int > conv_max_eig_ext) {
        conv_max_eig = conv_max_eig_int;
    } else {
        conv_max_eig = conv_max_eig_ext;
    }
   // conv_max_eig = std::max(conv_max_eig_int, conv_max_eig_ext);
   //Based on primitive jump
//   std::array<real,nstate> prim_int = pde_physics->convert_cons_to_prim(soln_int);
//   std::array<real,nstate> prim_ext = pde_physics->convert_cons_to_prim(soln_ext);
//   std::array<dealii::Tensor<1,dim,real>,nstate> grad_prim;
//   std::array<real,nstate> cons_avg;
//   for(int istate=0; istate<nstate; istate++){
//       for(int idim=0; idim<dim; idim++){
//           grad_prim[istate][idim] = prim_ext[istate] - prim_int[istate];
//       }
//       cons_avg[istate] = 0.5 * (soln_int[istate] + soln_ext[istate]);
//   }
//   std::array<dealii::Tensor<1,dim,real>,nstate> cons_grad = pde_physics->convert_grad_prim_to_grad_conservative(cons_avg, grad_prim);
//end based on primitive jump

//    std::array<real,nstate> prim_int = pde_physics->convert_cons_to_prim(soln_int);
//    std::array<real,nstate> prim_ext = pde_physics->convert_cons_to_prim(soln_ext);
//    std::array<real,nstate> cons_grad;
//    cons_grad[0] = prim_ext[0] - prim_int[0];
//    real kin_en_int = 0.0;
//    real kin_en_ext = 0.0;
//    for(int idim=0; idim<dim; idim++){
//        cons_grad[idim+1] = prim_ext[idim+1] * prim_ext[0] - prim_int[idim+1] * prim_int[0];
//        kin_en_int += prim_int[0] * prim_int[idim+1] * prim_int[idim+1];
//        kin_en_ext += prim_ext[0] * prim_ext[idim+1] * prim_ext[idim+1];
//    }
//    cons_grad[nstate-1] = (prim_ext[nstate-1] / 0.4 + 0.5 *kin_en_ext)  - (prim_int[nstate-1] / 0.4 + 0.5 *kin_en_int);

    std::array<real,nstate> entvar_int = pde_physics->compute_entropy_variables(soln_int);
    std::array<real,nstate> entvar_ext = pde_physics->compute_entropy_variables(soln_ext);
    real sign = 1.0;
    real sum = 0.0;
    for (int s=0; s<nstate; s++) {
        sum +=(entvar_ext[s] - entvar_int[s]) * (soln_ext[s] - soln_int[s]);
    }
    if(sum <0)
        sign = -1.0;
    // Scalar dissipation
    std::array<real, nstate> numerical_flux_dot_n;
    for (int s=0; s<nstate; s++) {
//        numerical_flux_dot_n[s] = - 0.5 * conv_max_eig * (soln_ext[s]-soln_int[s]);
//        numerical_flux_dot_n[s] = - 0.5 * conv_max_eig * cons_grad[s][0];//based on primitive jump
    //    numerical_flux_dot_n[s] = - 0.5 * conv_max_eig * cons_grad[s];
        //Entropy dissipative
        numerical_flux_dot_n[s] = - sign * 0.5 * conv_max_eig * (soln_ext[s]-soln_int[s]);
        //Entropy prod
//        numerical_flux_dot_n[s] = - 0.25 * abs(soln_ext[s]+soln_int[s]) * (soln_ext[s]-soln_int[s]);
//        numerical_flux_dot_n[s] -=  1.0/12.0 * abs(soln_ext[s]-soln_int[s]) * (soln_ext[s]-soln_int[s]);
    }

    return numerical_flux_dot_n;
}

template <int dim, int nstate, typename real>
void RoePikeRiemannSolverDissipation<dim,nstate,real>
::evaluate_entropy_fix (
    const std::array<real, 3> &eig_L,
    const std::array<real, 3> &eig_R,
    std::array<real, 3> &eig_RoeAvg,
    const real /*vel2_ravg*/,
    const real /*sound_ravg*/) const
{
    // Harten's entropy fix
    // -- See Blazek 2015, p.103-105
    for(int e=0;e<3;e++) {
        const real eps = std::max(abs(eig_RoeAvg[e] - eig_L[e]), abs(eig_R[e] - eig_RoeAvg[e]));
        if(eig_RoeAvg[e] < eps) {
            eig_RoeAvg[e] = 0.5*(eig_RoeAvg[e] * eig_RoeAvg[e]/eps + eps);
        }
    }
}

template <int dim, int nstate, typename real>
void RoePikeRiemannSolverDissipation<dim,nstate,real>
::evaluate_additional_modifications (
    const std::array<real, nstate> &/*soln_int*/,
    const std::array<real, nstate> &/*soln_ext*/,
    const std::array<real, 3> &/*eig_L*/,
    const std::array<real, 3> &/*eig_R*/,
    real &/*dV_normal*/, 
    dealii::Tensor<1,dim,real> &/*dV_tangent*/
    ) const
{
    // No additional modifications for the Roe-Pike scheme
}

template <int dim, int nstate, typename real>
void L2RoeRiemannSolverDissipation<dim,nstate,real>
::evaluate_shock_indicator (
    const std::array<real, 3> &eig_L,
    const std::array<real, 3> &eig_R,
    int &ssw_LEFT,
    int &ssw_RIGHT) const
{
    // Shock indicator of Wada & Liou (1994 Flux) -- Eq.(39)
    // -- See also p.74 of Osswald et al. (2016 L2Roe)
    
    ssw_LEFT=0; ssw_RIGHT=0; // initialize
    
    // ssw_L: i=L --> j=R
    if((eig_L[0]>0.0 && eig_R[0]<0.0) || (eig_L[2]>0.0 && eig_R[2]<0.0)) {
        ssw_LEFT = 1;
    }
    
    // ssw_R: i=R --> j=L
    if((eig_R[0]>0.0 && eig_L[0]<0.0) || (eig_R[2]>0.0 && eig_L[2]<0.0)) {
        ssw_RIGHT = 1;
    }
}

template <int dim, int nstate, typename real>
void L2RoeRiemannSolverDissipation<dim,nstate,real>
::evaluate_entropy_fix (
    const std::array<real, 3> &eig_L,
    const std::array<real, 3> &eig_R,
    std::array<real, 3> &eig_RoeAvg,
    const real vel2_ravg,
    const real sound_ravg) const
{
    // Van Leer et al. (1989 Sonic) entropy fix for acoustic waves
    // -- p.74 of Osswald et al. (2016 L2Roe)
    for(int e=0;e<3;e++) {
        if(e!=1) {
            // const real deig = std::max((eig_R[e]-eig_L[e]), 0.0);
            const real deig = std::max(static_cast<real>(eig_R[e] - eig_L[e]), static_cast<real>(0.0));
            if(eig_RoeAvg[e] < 2.0*deig) {
                eig_RoeAvg[e] = 0.25*(eig_RoeAvg[e] * eig_RoeAvg[e]/deig) + deig;
            }
        }
    }
    
    // Entropy fix of Liou (2000 Mass)
    // -- p.74 of Osswald et al. (2016 L2Roe)
    int ssw_L, ssw_R;
    evaluate_shock_indicator(eig_L,eig_R,ssw_L,ssw_R);
    if(ssw_L!=0 || ssw_R!=0) {
        eig_RoeAvg[1] = std::max(sound_ravg, static_cast<real>(sqrt(vel2_ravg)));
    }
}

template <int dim, int nstate, typename real>
void L2RoeRiemannSolverDissipation<dim,nstate,real>
::evaluate_additional_modifications  (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const std::array<real, 3> &eig_L,
    const std::array<real, 3> &eig_R,
    real &dV_normal, 
    dealii::Tensor<1,dim,real> &dV_tangent) const
{
    const real mach_number_L = this->euler_physics->compute_mach_number(soln_int);
    const real mach_number_R = this->euler_physics->compute_mach_number(soln_ext);

    // Osswald's two modifications to Roe-Pike scheme --> L2Roe
    // - Blending factor (variable 'z' in reference)
    const real blending_factor = std::min(static_cast<real>(1.0), std::max(mach_number_L,mach_number_R));
    // - Scale jump in (1) normal and (2) tangential velocities
    int ssw_L, ssw_R;
    evaluate_shock_indicator(eig_L,eig_R,ssw_L,ssw_R);
    if(ssw_L==0 && ssw_R==0)
    {
        dV_normal *= blending_factor;
        for (int d=0;d<dim;d++)
        {
            dV_tangent[d] *= blending_factor;
        }
    }
}

template <int dim, int nstate, typename real>
std::array<real, nstate> RoeBaseRiemannSolverDissipation<dim,nstate,real>
::evaluate_riemann_solver_dissipation (
    const std::array<real, nstate> &soln_int,
    const std::array<real, nstate> &soln_ext,
    const dealii::Tensor<1,dim,real> &normal_int) const
{
    #if 0
    //HLLC dissipation
// Using HLLC from Appendix B of Yu Lv and Matthias Ihme, 2014, Discontinuous Galerkin method for
    // multicomponent chemically reacting ﬂows and combustion.
    using RealArrayVector = std::array<dealii::Tensor<1,dim,real>,nstate>;

    const std::array<real,nstate> prim_soln_L = euler_physics->convert_conservative_to_primitive(soln_int);
    const std::array<real,nstate> prim_soln_R = euler_physics->convert_conservative_to_primitive(soln_ext);

    const real density_L = prim_soln_L[0];
    const real density_R = prim_soln_R[0];
    const real pressure_L = prim_soln_L[nstate-1];
    const real pressure_R = prim_soln_R[nstate-1];
    const dealii::Tensor< 1,dim,real > velocities_L = euler_physics->extract_velocities_from_primitive(prim_soln_L);
    const dealii::Tensor< 1,dim,real > velocities_R = euler_physics->extract_velocities_from_primitive(prim_soln_R);
    real velocity_dot_n_L = 0;
    real velocity_dot_n_R = 0;
    for(int d=0; d<dim; ++d)
    {
        velocity_dot_n_L += velocities_L[d]*normal_int[d];
        velocity_dot_n_R += velocities_R[d]*normal_int[d];
    }

    const real sound_L = euler_physics->compute_sound(density_L, pressure_L);
    const real sound_R = euler_physics->compute_sound(density_R, pressure_R);
/*
    // Pressure PVRS approach
    const real density_avg = 0.5*(density_L + density_R);
    const real sound_avg = 0.5*(sound_L + sound_R);
    const real pressure_pvrs = 0.5*(pressure_L + pressure_R) - 0.5*(velocity_dot_n_R - velocity_dot_n_L)*density_avg*sound_avg;
    real pressure_star = 0;
    if(pressure_pvrs > 0.0)
    {
        pressure_star = pressure_pvrs;
    }
*/

    // Hybrid scheme
    const real density_avg = 0.5*(density_L + density_R);
    const real sound_avg = 0.5*(sound_L + sound_R);
    const real pressure_pvrs = 0.5*(pressure_L + pressure_R) - 0.5*(velocity_dot_n_R - velocity_dot_n_L)*density_avg*sound_avg;
    real p_min = pressure_L;
    if(pressure_R<p_min)
    {
        p_min = pressure_R;
    }
    real pressure_star = 0;
    if(pressure_pvrs <= p_min)
    {
        // Two–Rarefaction Riemann solver TRRS approach
        const real zval = (euler_physics->gam - 1.0)/(2.0*euler_physics->gam);
        const real kernelval = (sound_L + sound_R - (euler_physics->gam-1.0)/2.0*(velocity_dot_n_R - velocity_dot_n_L))
                                / (sound_L/pow(pressure_L,zval) + sound_R/pow(pressure_R,zval));
        pressure_star = pow(kernelval,1.0/zval);
    }
    else
    {
        // Two–Shock Riemann solver TSRS approach
        real p0 = 0.0;
        if(pressure_pvrs>0.0)
        {
            p0 = pressure_pvrs;
        }
        const real A_L = 2.0/((euler_physics->gam+1)*density_L);
        const real A_R = 2.0/((euler_physics->gam+1)*density_R);
        const real B_L = (euler_physics->gam - 1.0)/(euler_physics->gam + 1.0)*pressure_L;
        const real B_R = (euler_physics->gam - 1.0)/(euler_physics->gam + 1.0)*pressure_R;
        const real g_L = sqrt(A_L/(p0 + B_L));
        const real g_R = sqrt(A_R/(p0 + B_R));
        pressure_star = (g_L*pressure_L + g_R*pressure_R - (velocity_dot_n_R - velocity_dot_n_L))/(g_L+g_R);
    }
    // Kernel of pressure-based approach.
    const real gam_fraction = (euler_physics->gam + 1.0)/(2.0*euler_physics->gam);
    real q_L = 1.0;
    if(pressure_star > pressure_L)
    {
        const real val = 1.0 + gam_fraction*(pressure_star/pressure_L - 1.0);
        q_L = sqrt(val);
    }
    real q_R = 1.0;
    if(pressure_star > pressure_R)
    {
        const real val = 1.0 + gam_fraction*(pressure_star/pressure_R - 1.0);
        q_R = sqrt(val);
    }

     real S_L = velocity_dot_n_L - sound_L*q_L;
     real S_R = velocity_dot_n_R + sound_R*q_R;

/*
    // Einfieldt's approach
    const real eta2 = 0.5 * sqrt(density_L)*sqrt(density_R)/(pow(sqrt(density_L) + sqrt(density_R),2));
    const real dbar_squared = (sqrt(density_L)*pow(sound_L,2) + sqrt(density_R)*pow(sound_R,2))/(sqrt(density_L) + sqrt(density_R))
                                + eta2*pow(velocity_dot_n_R - velocity_dot_n_L,2);
    const real dbar = sqrt(dbar_squared);
    const real ubar = (sqrt(density_L)*velocity_dot_n_L + sqrt(density_R)*velocity_dot_n_R)/(sqrt(density_L) + sqrt(density_R));
    real S_L = ubar - dbar;
    real S_R = ubar + dbar;
*/
/*
    if(use_upwinding)
    {
        // Simple Davis approach
        S_L = velocity_dot_n_L - sound_L;
        S_R = velocity_dot_n_R + sound_R;
    }
*/
/*
    // Using Roe based approaximations.
    (void) sound_L; (void) sound_R;
    const real ubar = (sqrt(density_L)*velocity_dot_n_L + sqrt(density_R)*velocity_dot_n_R)/(sqrt(density_L) + sqrt(density_R));
    const real enthalpy_bar = (sqrt(density_L)*euler_physics->compute_specific_enthalpy(soln_int, density_L)
                            + sqrt(density_R)*euler_physics->compute_specific_enthalpy(soln_ext, density_R))/(sqrt(density_L) + sqrt(density_R));
    const real sound_bar = sqrt(euler_physics->gamm1*(enthalpy_bar - 0.5*pow(ubar,2)));
    const real S_L = ubar - sound_bar;
    const real S_R = ubar + sound_bar;
*/

    const real S_star =
            (pressure_R - pressure_L + density_L*velocity_dot_n_L*(S_L - velocity_dot_n_L)
            - density_R*velocity_dot_n_R*(S_R - velocity_dot_n_R))/(density_L*(S_L - velocity_dot_n_L) - density_R*(S_R - velocity_dot_n_R));


    std::array<real, nstate> soln_star_L;
    std::array<real, nstate> soln_star_R;
    const real multfactor_L = (S_L - velocity_dot_n_L)/(S_L - S_star);
    const real multfactor_R = (S_R - velocity_dot_n_R)/(S_R - S_star);

    soln_star_L[0] = soln_int[0];
    soln_star_R[0] = soln_ext[0];

    for(int d=0; d<dim; ++d)
    {
        soln_star_L[1+d] = soln_int[1+d] + density_L*(S_star - velocity_dot_n_L)*normal_int[d];

        soln_star_R[1+d] = soln_ext[1+d] + density_R*(S_star - velocity_dot_n_R)*normal_int[d];
    }

    soln_star_L[nstate-1] = soln_int[nstate-1] + (S_star - velocity_dot_n_L)*(density_L*S_star + pressure_L/(S_L - velocity_dot_n_L));
    soln_star_R[nstate-1] = soln_ext[nstate-1] + (S_star - velocity_dot_n_R)*(density_R*S_star + pressure_R/(S_R - velocity_dot_n_R));

    for(int s=0; s<nstate; ++s)
    {
        soln_star_L[s] *= multfactor_L;
        soln_star_R[s] *= multfactor_R;
    }

    RealArrayVector conv_phys_flux_int;
    RealArrayVector conv_phys_flux_ext;

    conv_phys_flux_int = euler_physics->convective_flux (soln_int);
    conv_phys_flux_ext = euler_physics->convective_flux (soln_ext);


    std::array<real, nstate> numerical_flux_dot_n_L;
    std::array<real, nstate> numerical_flux_dot_n_R;

    for(int s = 0; s<nstate; ++s)
    {
        real flux_dot_n_L = 0.0;
        real flux_dot_n_R = 0.0;
        for(int d=0; d<dim; ++d)
        {
            flux_dot_n_L += conv_phys_flux_int[s][d]*normal_int[d];
            flux_dot_n_R += conv_phys_flux_ext[s][d]*normal_int[d];
        }
        numerical_flux_dot_n_L[s] = flux_dot_n_L;
        numerical_flux_dot_n_R[s] = flux_dot_n_R;
    }


    std::array<real, nstate> numerical_flux_dot_n;

    if( (S_L >= 0.0) )
    {
        for(int s=0; s<nstate; ++s)
        {
            numerical_flux_dot_n[s] = numerical_flux_dot_n_L[s];
        }
    }
    else if( (S_R <= 0.0) )
    {
        for(int s=0; s<nstate; ++s)
        {
            numerical_flux_dot_n[s] = numerical_flux_dot_n_R[s];
        }
    }
/*
    // Shock-stable modiﬁcation of the HLLC Riemann solver.
    else
    {
        const real Ma_limit = 0.1;
        const real Ma_local = std::max(abs(velocity_dot_n_L/sound_L), abs(velocity_dot_n_R/sound_R));
        const real pi = 4.0*atan(1.0);
        real min_val = 1.0;
        if((Ma_local/Ma_limit) < min_val)
        {
            min_val = (Ma_local/Ma_limit);
        }
        const real phi = sin(min_val*pi/2.0);
        const real S_L_lm = phi*S_L;
        const real S_R_lm = phi*S_R;

        for(int s=0; s<nstate; ++s)
        {
            numerical_flux_dot_n[s] = 0.5*(numerical_flux_dot_n_L[s] + numerical_flux_dot_n_R[s]) +
                0.5*(S_L_lm*(soln_star_L[s] - soln_int[s]) + abs(S_star)*(soln_star_L[s] - soln_star_R[s]) + S_R_lm*(soln_star_R[s] - soln_ext[s]));

        }
    }
*/


    else if( (S_L <= 0.0) && (0.0 < S_star))
    {
        for(int s=0; s<nstate; ++s)
        {
            numerical_flux_dot_n[s] = numerical_flux_dot_n_L[s] + S_L*(soln_star_L[s] - soln_int[s]);
        }
    }
    else if( (S_star <= 0.0) && (0.0 < S_R))
    {
        for(int s=0; s<nstate; ++s)
        {
            numerical_flux_dot_n[s] = numerical_flux_dot_n_R[s] + S_R*(soln_star_R[s] - soln_ext[s]);
        }
    }
    else
    {
        std::cout<<"Shouldn't have reached here in HLLC flux."<<std::endl;
        std::abort();
    }
    //difference central and entropy cons
    using RealArrayVector = std::array<dealii::Tensor<1,dim,real>,nstate>;
    RealArrayVector conv_phys_split_flux;
    conv_phys_split_flux = euler_physics->convective_numerical_split_flux (soln_int,soln_ext);
    RealArrayVector conv_phys_flux_int1;
    RealArrayVector conv_phys_flux_ext1;
    conv_phys_flux_int1 = euler_physics->convective_flux (soln_int);
    conv_phys_flux_ext1 = euler_physics->convective_flux (soln_ext);
    RealArrayVector flux_avg;
    for (int s=0; s<nstate; s++) {
        flux_avg[s] = 0.0;
        for (int d=0; d<dim; ++d) {
            flux_avg[s][d] = 0.5*(conv_phys_flux_int1[s][d] + conv_phys_flux_ext1[s][d]);
        }
    }
   // std::array<real,nstate> entropy_var_int = euler_physics->compute_entropy_variables(soln_int);
   // std::array<real,nstate> entropy_var_ext = euler_physics->compute_entropy_variables(soln_ext);
    for (int s=0; s<nstate; s++) {
    //    real flux_dot_n = 0.0;
        real flux_avg_dot_n = 0.0;
        for (int d=0; d<dim; ++d) {
     //       flux_dot_n += (flux_avg[s][d] - conv_phys_split_flux[s][d])*normal_int[d];
            flux_avg_dot_n += (flux_avg[s][d])*normal_int[d];
        }
//        numerical_flux_dot_n[s] -= abs((flux_dot_n)/((entropy_var_ext[s] - entropy_var_int[s]+1e-20)))*(entropy_var_ext[s] - entropy_var_int[s]);
        numerical_flux_dot_n[s] -= flux_avg_dot_n;
    }

    return numerical_flux_dot_n;

#endif


   // #if 0
//CUSP scheme
    const real pressure_int = euler_physics->compute_pressure(soln_int);
    const real pressure_ext = euler_physics->compute_pressure(soln_ext);
    const real specific_total_enthalpy_int = soln_int[nstate-1] / soln_int[0] + pressure_int / soln_int[0];
    const real specific_total_enthalpy_ext = soln_ext[nstate-1] / soln_ext[0] + pressure_ext / soln_ext[0];

    //Compute Roe averages
    dealii::Tensor<1,dim,real> vel_roe_avg;
    real vel_roe_avg_sqr = 0.0;
    real contravariant_vel = 0.0;
    real vel_R = 0.0;
    real vel_L = 0.0;
    for(int idim=0; idim<dim; idim++){
        vel_roe_avg[idim] = (soln_int[idim+1]/soln_int[0]*sqrt(soln_int[0])
                          + soln_ext[idim+1]/soln_ext[0]*sqrt(soln_ext[0]))
                          / (sqrt(soln_int[0]) + sqrt(soln_ext[0]));
        vel_roe_avg_sqr += vel_roe_avg[idim] * vel_roe_avg[idim];
        contravariant_vel += vel_roe_avg[idim] * normal_int[idim];
        vel_R += soln_ext[idim+1] / soln_ext[0] * normal_int[idim];
        vel_L += soln_int[idim+1] / soln_int[0] * normal_int[idim];
    }
    const real enthalpy_roe_avg = (specific_total_enthalpy_int*sqrt(soln_int[0])
                                + specific_total_enthalpy_ext*sqrt(soln_ext[0]))
                                / (sqrt(soln_int[0]) + sqrt(soln_ext[0]));
    const real speed_sound = sqrt(euler_physics->gamm1 * (enthalpy_roe_avg - 0.5 * vel_roe_avg_sqr));
    const real gamma_minus = (euler_physics->gam + 1.0) / (2.0*euler_physics->gam) * contravariant_vel
                           - sqrt( (euler_physics->gamm1/(2.0*euler_physics->gam)*contravariant_vel) *(euler_physics->gamm1/(2.0*euler_physics->gam)*contravariant_vel) + speed_sound * speed_sound / euler_physics->gam );
    const real gamma_plus = (euler_physics->gam + 1.0) / (2.0*euler_physics->gam) * contravariant_vel
                           + sqrt( (euler_physics->gamm1/(2.0*euler_physics->gam)*contravariant_vel) *(euler_physics->gamm1/(2.0*euler_physics->gam)*contravariant_vel) + speed_sound * speed_sound / euler_physics->gam );
    const real mach_number = contravariant_vel / speed_sound;
    real beta = 0.0;
    if(mach_number < 1.0 && mach_number >= 0.0) {
        real val = (contravariant_vel + gamma_minus) / (contravariant_vel - gamma_minus);
        if(val > 0.0)
            beta = val;
        else
            beta = 0.0;
    }
    else if(mach_number < 0.0 && mach_number >= -1){
        real val = (contravariant_vel + gamma_plus) / (contravariant_vel - gamma_plus);
        if(val > 0.0)
            beta = - val;
        else
            beta = 0.0;
    }
    else if(mach_number >= 1.0)
        beta = 1.0;
    else if(mach_number <= -1.0)
        beta = -1.0;


    real alpha_c = 0.0;
    if(abs(beta) <= 1e-14)
        alpha_c = abs(contravariant_vel);
    else if (beta > 0.0 && 0.0 < mach_number && mach_number < 1.0)
        alpha_c = - (1.0 + beta) * gamma_minus;
    else if (beta < 0.0 && -1.0 < mach_number && mach_number < 0.0)
        alpha_c =  (1.0 - beta) * gamma_plus;
    else if (abs(mach_number) >= 1.0)
        alpha_c = 0.0;


    std::array<real,nstate> dissipation;
    for(int istate=0;istate<nstate; istate++){
        const real u_L = (istate == nstate-1) ? soln_int[0] * specific_total_enthalpy_int
                  : soln_int[istate];
        const real u_R = (istate == nstate-1) ? soln_ext[0] * specific_total_enthalpy_ext
                  : soln_ext[istate];
        dissipation[istate] = - 0.5 * alpha_c * (u_R - u_L);
        dissipation[istate] -= 0.5 * beta *(u_R * vel_R - u_L * vel_L);
        if(istate > 0 && istate < nstate - 1){//momentum equations add pressure
            dissipation[istate] -= 0.5 * beta * (pressure_ext * normal_int[istate-1]
                                 - pressure_int * normal_int[istate-1]);
        }

    }
//    //difference central and entropy cons
//    using RealArrayVector = std::array<dealii::Tensor<1,dim,real>,nstate>;
//    RealArrayVector conv_phys_split_flux;
//    conv_phys_split_flux = euler_physics->convective_numerical_split_flux (soln_int,soln_ext);
//    RealArrayVector conv_phys_flux_int;
//    RealArrayVector conv_phys_flux_ext;
//    conv_phys_flux_int = euler_physics->convective_flux (soln_int);
//    conv_phys_flux_ext = euler_physics->convective_flux (soln_ext);
//    RealArrayVector flux_avg;
//    for (int s=0; s<nstate; s++) {
//        flux_avg[s] = 0.0;
//        for (int d=0; d<dim; ++d) {
//            flux_avg[s][d] = 0.5*(conv_phys_flux_int[s][d] + conv_phys_flux_ext[s][d]);
//        }
//    }
//    std::array<real,nstate> entropy_var_int = euler_physics->compute_entropy_variables(soln_int);
//    std::array<real,nstate> entropy_var_ext = euler_physics->compute_entropy_variables(soln_ext);
//    for (int s=0; s<nstate; s++) {
//        real flux_dot_n = 0.0;
//        for (int d=0; d<dim; ++d) {
//            flux_dot_n += (flux_avg[s][d] - conv_phys_split_flux[s][d])*normal_int[d];
//        }
//        dissipation[s] -= abs((flux_dot_n)/((entropy_var_ext[s] - entropy_var_int[s]+1e-20)))*(entropy_var_ext[s] - entropy_var_int[s]);
//    }

#if 0

    //entropy dissipative Roe scheme
    std::array<std::array<real,nstate>,nstate> eigenvector_matrix;
    std::array<real,nstate> eigenvalues;
    std::array<real,nstate> eigenvalue_scale;

    const std::array<real,nstate> primitive_soln_int = euler_physics->convert_conservative_to_primitive(soln_int);
    const std::array<real,nstate> primitive_soln_ext = euler_physics->convert_conservative_to_primitive(soln_ext);
    //The IR and Chandrashekar dissipative fluxes with production
    const std::array<real,nstate> roe_var1 = euler_physics->compute_ismail_roe_parameter_vector_from_primitive(primitive_soln_int);
    const std::array<real,nstate> roe_var2 = euler_physics->compute_ismail_roe_parameter_vector_from_primitive(primitive_soln_ext);

    // Compute mean (average) parameter vector
    std::array<real,nstate> avg_roe_var;
    for(int s=0; s<nstate; ++s){
        avg_roe_var[s] = 0.5*(roe_var1[s] + roe_var2[s]);
    }

    // Compute logarithmic mean parameter vector
    std::array<real,nstate> log_mean_roe_var;
    for(int s=0; s<nstate; ++s){
        log_mean_roe_var[s] = euler_physics->compute_ismail_roe_logarithmic_mean(roe_var1[s], roe_var2[s]);
    }

    std::array<real,nstate> roe_avg_var;
    roe_avg_var[0] = avg_roe_var[0] * log_mean_roe_var[nstate-1];
    real roe_avg_vel_sqr = 0.0;
    real roe_avg_vel_dot_n = 0.0;
    real vel_dot_n_L = 0.0;
    real vel_dot_n_R = 0.0;
    real vel_sqr_L = 0.0;
    real vel_sqr_R = 0.0;


    //compute using roe average
    roe_avg_var[0] = sqrt(primitive_soln_int[0]*primitive_soln_ext[0]);
    for(int idim=0; idim<dim; idim++){
        roe_avg_var[idim+1] = (sqrt(primitive_soln_int[0])*primitive_soln_int[idim+1] + sqrt(primitive_soln_ext[0])*primitive_soln_ext[idim+1])
                            /(sqrt(primitive_soln_int[0]) + sqrt(primitive_soln_ext[0]));
        roe_avg_vel_dot_n += roe_avg_var[idim+1] * normal_int[idim];
        roe_avg_vel_sqr += roe_avg_var[idim+1] * roe_avg_var[idim+1];
        vel_dot_n_L += primitive_soln_int[idim+1] * normal_int[idim];
        vel_dot_n_R += primitive_soln_ext[idim+1] * normal_int[idim];
        vel_sqr_L += primitive_soln_int[idim+1] * primitive_soln_int[idim+1];
        vel_sqr_R += primitive_soln_ext[idim+1] * primitive_soln_ext[idim+1];
    }
    const real specific_enthalpy_L = euler_physics->compute_specific_enthalpy(soln_int, primitive_soln_int[nstate-1]);
    const real specific_enthalpy_R = euler_physics->compute_specific_enthalpy(soln_ext, primitive_soln_ext[nstate-1]);
    const real a_L = sqrt(euler_physics->gamm1 * (specific_enthalpy_L - 0.5 * vel_sqr_L));
    const real a_R = sqrt(euler_physics->gamm1 * (specific_enthalpy_R - 0.5 * vel_sqr_R));
    roe_avg_var[nstate-1] = (sqrt(primitive_soln_int[0])*specific_enthalpy_L + sqrt(primitive_soln_ext[0])*specific_enthalpy_R)
                            /(sqrt(primitive_soln_int[0]) + sqrt(primitive_soln_ext[0]));
    real a = sqrt(euler_physics->gamm1 *(roe_avg_var[nstate-1] - 0.5 * roe_avg_vel_sqr));
    real p1 = (sqrt(primitive_soln_int[0])*primitive_soln_int[nstate-1] + sqrt(primitive_soln_ext[0])*primitive_soln_ext[nstate-1])
                            /(sqrt(primitive_soln_int[0]) + sqrt(primitive_soln_ext[0]));

    //build eigenvector matrices
    //first column
    eigenvector_matrix[0][0] = 1.0;
    for(int jdim=0; jdim<dim; jdim++){
        eigenvector_matrix[jdim+1][0] = roe_avg_var[jdim+1] - a * normal_int[jdim];
    }
    eigenvector_matrix[nstate-1][0] = roe_avg_var[nstate-1] - roe_avg_vel_dot_n * a;
    //second column
    eigenvector_matrix[0][1] = 1.0;
    for(int jdim=0; jdim<dim; jdim++){
        eigenvector_matrix[jdim+1][1] = roe_avg_var[jdim+1];
    }
    eigenvector_matrix[nstate-1][1] = 0.5 * roe_avg_vel_sqr;
    //3-4 columns if 2D or 3D
    if constexpr(dim>=2){
        eigenvector_matrix[0][2] = 0.0;
        eigenvector_matrix[1][2] = normal_int[1];
        eigenvector_matrix[2][2] = - normal_int[0];
        if constexpr(dim == 3) eigenvector_matrix[3][2] = 0.0;
        eigenvector_matrix[nstate-1][2] = roe_avg_var[1] * normal_int[1] - roe_avg_var[2] * normal_int[0];
    }
    if constexpr(dim==3){
        eigenvector_matrix[0][3] = 0.0;
        eigenvector_matrix[1][3] = - normal_int[2];
        eigenvector_matrix[2][3] = 0.0;
        eigenvector_matrix[3][3] = - normal_int[0];
        eigenvector_matrix[nstate-1][3] = roe_avg_var[2] * normal_int[0] - roe_avg_var[1] * normal_int[2];
    }
    //last column
    eigenvector_matrix[0][nstate-1] = 1.0;
    for(int jdim=0; jdim<dim; jdim++){
        eigenvector_matrix[jdim+1][nstate-1] = roe_avg_var[jdim+1] + a * normal_int[jdim];
    }
    eigenvector_matrix[nstate-1][nstate-1] = roe_avg_var[nstate-1] + roe_avg_vel_dot_n * a;

    //entropy production
    eigenvalues[0] = 1.0/6.0 * abs((vel_dot_n_R - a_R) - (vel_dot_n_L - a_L));
    eigenvalues[nstate-1] = 1.0/6.0 * abs((vel_dot_n_R + a_R) - (vel_dot_n_L + a_L));

    for(int jdim=0; jdim<dim; jdim++){
        eigenvalues[jdim+1] = 0.0;
    }
    //build eigenvalue scale vector
    eigenvalue_scale[0] = roe_avg_var[0] / (2.0*euler_physics->gam);
    eigenvalue_scale[1] = euler_physics->gamm1 * roe_avg_var[0] / euler_physics->gam;
    eigenvalue_scale[nstate-1] = roe_avg_var[0] / (2.0*euler_physics->gam);
    for(int idim=1; idim<dim; idim++){//only makes difference 2D and 3D
        eigenvalue_scale[idim+1] = p1;
    }

    std::array<std::array<real,nstate>,nstate> dissipation_matrix;
    for(int istate=0; istate<nstate; istate++){
        for(int jstate=0; jstate<nstate; jstate++){
            dissipation_matrix[istate][jstate] = 0.0;
            for(int kstate=0; kstate<nstate; kstate++){
                dissipation_matrix[istate][jstate] += eigenvector_matrix[istate][kstate] //R
                                                    * eigenvector_matrix[jstate][kstate] //R^T
                                                    * eigenvalues[kstate]
                                                    * eigenvalue_scale[kstate];
            }
        }
    }

    std::array<real,nstate> entropy_var_int = euler_physics->compute_entropy_variables(soln_int);
    std::array<real,nstate> entropy_var_ext = euler_physics->compute_entropy_variables(soln_ext);

    for(int istate=0;istate<nstate; istate++){
        for(int jstate=0; jstate<nstate; jstate++){
            dissipation[istate] -= 0.5 * dissipation_matrix[istate][jstate]
                                 * (entropy_var_ext[jstate] - entropy_var_int[jstate]);
        }
    }
#endif

    return dissipation;
   // #endif


#if 0
    //entropy dissipative Roe scheme
    std::array<std::array<real,nstate>,nstate> eigenvector_matrix;
    std::array<real,nstate> eigenvalues;
//    std::array<real,nstate> eigenvalues_entropy_prod;
    std::array<real,nstate> eigenvalue_scale;

    const std::array<real,nstate> primitive_soln_int = euler_physics->convert_conservative_to_primitive(soln_int);
    const std::array<real,nstate> primitive_soln_ext = euler_physics->convert_conservative_to_primitive(soln_ext);
    //The IR and Chandrashekar dissipative fluxes with production
    const std::array<real,nstate> roe_var1 = euler_physics->compute_ismail_roe_parameter_vector_from_primitive(primitive_soln_int);
    const std::array<real,nstate> roe_var2 = euler_physics->compute_ismail_roe_parameter_vector_from_primitive(primitive_soln_ext);

    // Compute mean (average) parameter vector
    std::array<real,nstate> avg_roe_var;
    for(int s=0; s<nstate; ++s){
        avg_roe_var[s] = 0.5*(roe_var1[s] + roe_var2[s]);
    }

    // Compute logarithmic mean parameter vector
    std::array<real,nstate> log_mean_roe_var;
    for(int s=0; s<nstate; ++s){
        log_mean_roe_var[s] = euler_physics->compute_ismail_roe_logarithmic_mean(roe_var1[s], roe_var2[s]);
    }

    std::array<real,nstate> roe_avg_var;
    roe_avg_var[0] = avg_roe_var[0] * log_mean_roe_var[nstate-1];
    real roe_avg_vel_sqr = 0.0;
    real roe_avg_vel_dot_n = 0.0;
    real vel_dot_n_L = 0.0;
    real vel_dot_n_R = 0.0;
    real vel_sqr_L = 0.0;
    real vel_sqr_R = 0.0;
    for(int idim=0; idim<dim; idim++){
        roe_avg_var[idim+1] = avg_roe_var[idim+1] / avg_roe_var[0];
        roe_avg_vel_sqr += roe_avg_var[idim+1] * roe_avg_var[idim+1];
        roe_avg_vel_dot_n += roe_avg_var[idim+1] * normal_int[idim];
        vel_dot_n_L += primitive_soln_int[idim+1] * normal_int[idim];
        vel_dot_n_R += primitive_soln_ext[idim+1] * normal_int[idim];
        vel_sqr_L += primitive_soln_int[idim+1] * primitive_soln_int[idim+1];
        vel_sqr_R += primitive_soln_ext[idim+1] * primitive_soln_ext[idim+1];
    }
    real p1 = avg_roe_var[nstate-1] / avg_roe_var[0];
    real p2 = (euler_physics->gam +1.0)/(euler_physics->gam*2.0) * log_mean_roe_var[nstate-1] / log_mean_roe_var[0]
            + (euler_physics->gamm1)/(euler_physics->gam * 2.0) * p1;
    real a = sqrt(euler_physics->gam * p2 / roe_avg_var[0]);
    roe_avg_var[nstate-1] = a * a / euler_physics->gamm1 + 0.5 * roe_avg_vel_sqr;

    const real specific_enthalpy_L = euler_physics->compute_specific_enthalpy(soln_int, primitive_soln_int[nstate-1]);
    const real specific_enthalpy_R = euler_physics->compute_specific_enthalpy(soln_ext, primitive_soln_ext[nstate-1]);
    const real a_L = sqrt(euler_physics->gamm1 * (specific_enthalpy_L - 0.5 * vel_sqr_L));
    const real a_R = sqrt(euler_physics->gamm1 * (specific_enthalpy_R - 0.5 * vel_sqr_R));

//#if 0
    //compute using roe average
    roe_avg_var[0] = sqrt(primitive_soln_int[0]*primitive_soln_ext[0]);
    for(int idim=0; idim<dim; idim++){
        roe_avg_var[idim+1] = (sqrt(primitive_soln_int[0])*primitive_soln_int[idim+1] + sqrt(primitive_soln_ext[0])*primitive_soln_ext[idim+1])
                            /(sqrt(primitive_soln_int[0]) + sqrt(primitive_soln_ext[0]));
        roe_avg_vel_dot_n += roe_avg_var[idim+1] * normal_int[idim];
        roe_avg_vel_sqr += roe_avg_var[idim+1] * roe_avg_var[idim+1];
        vel_dot_n_L += primitive_soln_int[idim+1] * normal_int[idim];
        vel_dot_n_R += primitive_soln_ext[idim+1] * normal_int[idim];
        vel_sqr_L += primitive_soln_int[idim+1] * primitive_soln_int[idim+1];
        vel_sqr_R += primitive_soln_ext[idim+1] * primitive_soln_ext[idim+1];
    }
    roe_avg_var[nstate-1] = (sqrt(primitive_soln_int[0])*specific_enthalpy_L + sqrt(primitive_soln_ext[0])*specific_enthalpy_R)
                            /(sqrt(primitive_soln_int[0]) + sqrt(primitive_soln_ext[0]));
    a = sqrt(euler_physics->gamm1 *(roe_avg_var[nstate-1] - 0.5 * roe_avg_vel_sqr));
//#endif

    //build eigenvector matrices
    //first column
    eigenvector_matrix[0][0] = 1.0;
    for(int jdim=0; jdim<dim; jdim++){
        eigenvector_matrix[jdim+1][0] = roe_avg_var[jdim+1] - a * normal_int[jdim];
    }
    eigenvector_matrix[nstate-1][0] = roe_avg_var[nstate-1] - roe_avg_vel_dot_n * a;
    //second column
    eigenvector_matrix[0][1] = 1.0;
    for(int jdim=0; jdim<dim; jdim++){
        eigenvector_matrix[jdim+1][1] = roe_avg_var[jdim+1];
    }
    eigenvector_matrix[nstate-1][1] = 0.5 * roe_avg_vel_sqr;
    //3-4 columns if 2D or 3D
    if constexpr(dim>=2){
        eigenvector_matrix[0][2] = 0.0;
        eigenvector_matrix[1][2] = normal_int[1];
        eigenvector_matrix[2][2] = - normal_int[0];
        if constexpr(dim == 3) eigenvector_matrix[3][2] = 0.0;
        eigenvector_matrix[nstate-1][2] = roe_avg_var[1] * normal_int[1] - roe_avg_var[2] * normal_int[0];
    }
    if constexpr(dim==3){
        eigenvector_matrix[0][3] = 0.0;
        eigenvector_matrix[1][3] = - normal_int[2];
        eigenvector_matrix[2][3] = 0.0;
        eigenvector_matrix[3][3] = - normal_int[0];
        eigenvector_matrix[nstate-1][3] = roe_avg_var[2] * normal_int[0] - roe_avg_var[1] * normal_int[2];
    }
    //last column
    eigenvector_matrix[0][nstate-1] = 1.0;
    for(int jdim=0; jdim<dim; jdim++){
        eigenvector_matrix[jdim+1][nstate-1] = roe_avg_var[jdim+1] + a * normal_int[jdim];
    }
    eigenvector_matrix[nstate-1][nstate-1] = roe_avg_var[nstate-1] + roe_avg_vel_dot_n * a;
    //build eigenvalue vector
    eigenvalues[0] = abs(roe_avg_vel_dot_n - a);
    real lambda_max = eigenvalues[0];
    eigenvalues[nstate-1] = abs(roe_avg_vel_dot_n + a);
    if(eigenvalues[nstate-1] > lambda_max)
        lambda_max = eigenvalues[nstate-1];


    for(int jdim=0; jdim<dim; jdim++){
        eigenvalues[jdim+1] = abs(roe_avg_vel_dot_n);
   //     eigenvalues_entropy_prod[jdim+1] = 0.0;
     //   eigenvalues[jdim+1] += abs(vel_dot_n_R - vel_dot_n_L);//Idea
        if(eigenvalues[jdim+1] > lambda_max)
            lambda_max = eigenvalues[jdim+1];
    }
    //Harten's entropy fix
#if 0
    real delta = 1.0/10.0 * lambda_max;
    for(int istate=0; istate<nstate; istate++){
        if(eigenvalues[istate] < delta)
            eigenvalues[istate] = (eigenvalues[istate]*eigenvalues[istate]+delta*delta)/(2.0*delta);
    }
#endif

    //entropy production
  //  eigenvalues_entropy_prod[0] = 1.0/6.0 * abs((vel_dot_n_R - a_R) - (vel_dot_n_L - a_L));
  //  eigenvalues_entropy_prod[nstate-1] = 1.0/6.0 * abs((vel_dot_n_R + a_R) - (vel_dot_n_L + a_L));
    eigenvalues[0] += 1.0/6.0 * abs((vel_dot_n_R - a_R) - (vel_dot_n_L - a_L));
    eigenvalues[nstate-1] += 1.0/6.0 * abs((vel_dot_n_R + a_R) - (vel_dot_n_L + a_L));

    //build eigenvalue scale vector
    eigenvalue_scale[0] = roe_avg_var[0] / (2.0*euler_physics->gam);
    eigenvalue_scale[1] = euler_physics->gamm1 * roe_avg_var[0] / euler_physics->gam;
    eigenvalue_scale[nstate-1] = roe_avg_var[0] / (2.0*euler_physics->gam);
    for(int idim=1; idim<dim; idim++){//only makes difference 2D and 3D
        eigenvalue_scale[idim+1] = p1;
    }

    std::array<std::array<real,nstate>,nstate> dissipation_matrix;
  //  std::array<std::array<real,nstate>,nstate> dissipation_matrix_prod;
    for(int istate=0; istate<nstate; istate++){
        for(int jstate=0; jstate<nstate; jstate++){
            dissipation_matrix[istate][jstate] = 0.0;
  //          dissipation_matrix_prod[istate][jstate] = 0.0;
            for(int kstate=0; kstate<nstate; kstate++){
                dissipation_matrix[istate][jstate] += eigenvector_matrix[istate][kstate] //R
                                                    * eigenvector_matrix[jstate][kstate] //R^T
                                                    * eigenvalues[kstate]
                                                 //   * eigenvalues_entropy_prod[kstate]
                                                    * eigenvalue_scale[kstate];
      //          dissipation_matrix_prod[istate][jstate] += eigenvector_matrix[istate][kstate] //R
      //                                              * eigenvector_matrix[jstate][kstate] //R^T
      //                                              * eigenvalues_entropy_prod[kstate]
      //                                              * eigenvalue_scale[kstate];
            }
        }
    }

    std::array<real,nstate> entropy_var_int = euler_physics->compute_entropy_variables(soln_int);
    std::array<real,nstate> entropy_var_ext = euler_physics->compute_entropy_variables(soln_ext);

    std::array<real,nstate> dissipation;
//    std::array<real,nstate> dissipation_prod;
    for(int istate=0;istate<nstate; istate++){
        dissipation[istate] = 0.0;
 //       dissipation_prod[istate] = 0.0;
        for(int jstate=0; jstate<nstate; jstate++){
            dissipation[istate] -= 0.5 * dissipation_matrix[istate][jstate]
                              //   * (entropy_var_ext[jstate] - entropy_var_int[jstate])
                              //   * euler_physics->gamm1;
                                 * (entropy_var_ext[jstate] - entropy_var_int[jstate]);
    //        dissipation_prod[istate] -= 0.5 * dissipation_matrix_prod[istate][jstate]
    //                             * (entropy_var_ext[jstate] - entropy_var_int[jstate]);
        }
    }

//    //difference central and entropy cons
//    using RealArrayVector = std::array<dealii::Tensor<1,dim,real>,nstate>;
//    RealArrayVector conv_phys_split_flux;
//    conv_phys_split_flux = euler_physics->convective_numerical_split_flux (soln_int,soln_ext);
//    RealArrayVector conv_phys_flux_int;
//    RealArrayVector conv_phys_flux_ext;
//    conv_phys_flux_int = euler_physics->convective_flux (soln_int);
//    conv_phys_flux_ext = euler_physics->convective_flux (soln_ext);
//    RealArrayVector flux_avg;
//    for (int s=0; s<nstate; s++) {
//        flux_avg[s] = 0.0;
//        for (int d=0; d<dim; ++d) {
//            flux_avg[s][d] = 0.5*(conv_phys_flux_int[s][d] + conv_phys_flux_ext[s][d]);
//        }
//    }
//    std::array<real,nstate> dissipation_prod_2;
//    for (int s=0; s<nstate; s++) {
//        dissipation_prod_2[s] =0.0;
//        real flux_dot_n = 0.0;
//        for (int d=0; d<dim; ++d) {
//            flux_dot_n += (flux_avg[s][d] - conv_phys_split_flux[s][d])*normal_int[d];
//        }
////        dissipation[s] -= abs((flux_dot_n)/((entropy_var_ext[s] - entropy_var_int[s]+1e-20)))*(entropy_var_ext[s] - entropy_var_int[s]);
//        dissipation_prod_2[s] -= abs((flux_dot_n)/((entropy_var_ext[s] - entropy_var_int[s]+1e-20)))*(entropy_var_ext[s] - entropy_var_int[s]);
//    }
//    real entropy_prod1=0.0;
//    real entropy_prod2=0.0;
//    for (int s=0; s<nstate; s++) {
//        entropy_prod1 += dissipation_prod[s]  *(entropy_var_ext[s] - entropy_var_int[s]);
//        entropy_prod2 += dissipation_prod_2[s]  *(entropy_var_ext[s] - entropy_var_int[s]);
//    }
//    if(abs(entropy_prod1-entropy_prod2)>1e-8)
//        std::cout<<"difference "<<entropy_prod1-entropy_prod2<<" prod first "<<entropy_prod1<<" vs "<<entropy_prod2<<std::endl;



    return dissipation;
#endif



#if 0

    std::array<std::array<real,nstate>,nstate> eigenvector_matrix;
    std::array<real,nstate> eigenvalues;
    std::array<real,nstate> eigenvalue_scale;

    const std::array<real,nstate> primitive_soln_int = euler_physics->convert_conservative_to_primitive(soln_int);
    const std::array<real,nstate> primitive_soln_ext = euler_physics->convert_conservative_to_primitive(soln_ext);
    //Changing the definition for the averages used in dissipation to be consistent with the entorpy conserving and kinetic energy preserving flux to see difference

    const real rho_log = euler_physics->compute_ismail_roe_logarithmic_mean(soln_int[0], soln_ext[0]);
    const real pressure1 = euler_physics->compute_pressure(soln_int);
    const real pressure2 = euler_physics->compute_pressure(soln_ext);

    const real beta1 = soln_int[0]/(pressure1);
    const real beta2 = soln_ext[0]/(pressure2);

    const real beta_log = euler_physics->compute_ismail_roe_logarithmic_mean(beta1, beta2);
    const dealii::Tensor<1,dim,real> vel1 = euler_physics->compute_velocities(soln_int);
    const dealii::Tensor<1,dim,real> vel2 = euler_physics->compute_velocities(soln_ext);

    const real pressure_hat = 0.5*(pressure1+pressure2);

    dealii::Tensor<1,dim,real> vel_avg;
    real vel_square_avg = 0.0;;
    for(int idim=0; idim<dim; idim++){
        vel_avg[idim] = 0.5*(vel1[idim]+vel2[idim]);
        vel_square_avg += (0.5 *(vel1[idim]+vel2[idim])) * (0.5 *(vel1[idim]+vel2[idim]));
    }

    //enthalpy hat from Chandrashekar
    real enthalpy_hat = 1.0/(2.0*beta_log*euler_physics->gamm1) + vel_square_avg + pressure_hat/rho_log;

    for(int idim=0; idim<dim; idim++){
        enthalpy_hat -= 0.5*(0.5*(vel1[idim]*vel1[idim] + vel2[idim]*vel2[idim]));
    }


    std::array<real,nstate> roe_avg_var;
    roe_avg_var[0] = rho_log;
    real roe_avg_vel_sqr = 0.0;
    real roe_avg_vel_dot_n = 0.0;
    real vel_dot_n_L = 0.0;
    real vel_dot_n_R = 0.0;
    real vel_sqr_L = 0.0;
    real vel_sqr_R = 0.0;
    for(int idim=0; idim<dim; idim++){
        roe_avg_var[idim+1] = vel_avg[idim];
        roe_avg_vel_sqr += roe_avg_var[idim+1] * roe_avg_var[idim+1];
        roe_avg_vel_dot_n += roe_avg_var[idim+1] * normal_int[idim];
        vel_dot_n_L += primitive_soln_int[idim+1] * normal_int[idim];
        vel_dot_n_R += primitive_soln_ext[idim+1] * normal_int[idim];
        vel_sqr_L += primitive_soln_int[idim+1] * primitive_soln_int[idim+1];
        vel_sqr_R += primitive_soln_ext[idim+1] * primitive_soln_ext[idim+1];
    }
    roe_avg_var[nstate-1] = enthalpy_hat;
    const real p1 = pressure_hat;

    const real a = sqrt(euler_physics->gamm1 *(enthalpy_hat - 0.5 * roe_avg_vel_sqr));

    const real specific_enthalpy_L = euler_physics->compute_specific_enthalpy(soln_int, primitive_soln_int[nstate-1]);
    const real specific_enthalpy_R = euler_physics->compute_specific_enthalpy(soln_ext, primitive_soln_ext[nstate-1]);
    const real a_L = sqrt(euler_physics->gamm1 * (specific_enthalpy_L - 0.5 * vel_sqr_L));
    const real a_R = sqrt(euler_physics->gamm1 * (specific_enthalpy_R - 0.5 * vel_sqr_R));

    //build eigenvector matrices
    //first column
    eigenvector_matrix[0][0] = 1.0;
    for(int jdim=0; jdim<dim; jdim++){
        eigenvector_matrix[jdim+1][0] = roe_avg_var[jdim+1] - a * normal_int[jdim];
    }
    eigenvector_matrix[nstate-1][0] = roe_avg_var[nstate-1] - roe_avg_vel_dot_n * a;
    //second column
    eigenvector_matrix[0][1] = 1.0;
    for(int jdim=0; jdim<dim; jdim++){
        eigenvector_matrix[jdim+1][1] = roe_avg_var[jdim+1];
    }
    eigenvector_matrix[nstate-1][1] = 0.5 * roe_avg_vel_sqr;
    //3-4 columns if 2D or 3D
    if constexpr(dim>=2){
        eigenvector_matrix[0][2] = 0.0;
        eigenvector_matrix[1][2] = normal_int[1];
        eigenvector_matrix[2][2] = - normal_int[0];
        if constexpr(dim == 3) eigenvector_matrix[3][2] = 0.0;
        eigenvector_matrix[nstate-1][2] = roe_avg_var[1] * normal_int[1] - roe_avg_var[2] * normal_int[0];
    }
    if constexpr(dim==3){
        eigenvector_matrix[0][3] = 0.0;
        eigenvector_matrix[1][3] = - normal_int[2];
        eigenvector_matrix[2][3] = 0.0;
        eigenvector_matrix[3][3] = - normal_int[0];
        eigenvector_matrix[nstate-1][3] = roe_avg_var[2] * normal_int[0] - roe_avg_var[1] * normal_int[2];
    }
    //last column
    eigenvector_matrix[0][nstate-1] = 1.0;
    for(int jdim=0; jdim<dim; jdim++){
        eigenvector_matrix[jdim+1][nstate-1] = roe_avg_var[jdim+1] + a * normal_int[jdim];
    }
    eigenvector_matrix[nstate-1][nstate-1] = roe_avg_var[nstate-1] + roe_avg_vel_dot_n * a;
    //build eigenvalue vector
    eigenvalues[0] = abs(roe_avg_vel_dot_n - a);
    eigenvalues[nstate-1] = abs(roe_avg_vel_dot_n + a);

    eigenvalues[0] += 1.0/6.0 * abs((vel_dot_n_R - a_R) - (vel_dot_n_L - a_L));
    eigenvalues[nstate-1] += 1.0/6.0 * abs((vel_dot_n_R + a_R) - (vel_dot_n_L + a_L));

    for(int jdim=0; jdim<dim; jdim++){
        eigenvalues[jdim+1] = abs(roe_avg_vel_dot_n);
    }
    //build eigenvalue scale vector
    eigenvalue_scale[0] = roe_avg_var[0] / (2.0*euler_physics->gam);
    eigenvalue_scale[1] = euler_physics->gamm1 * roe_avg_var[0] / euler_physics->gam;
    eigenvalue_scale[nstate-1] = roe_avg_var[0] / (2.0*euler_physics->gam);
    for(int idim=1; idim<dim; idim++){//only makes difference 2D and 3D
        eigenvalue_scale[idim+1] = p1;
    }

    std::array<std::array<real,nstate>,nstate> dissipation_matrix;
    for(int istate=0; istate<nstate; istate++){
        for(int jstate=0; jstate<nstate; jstate++){
            dissipation_matrix[istate][jstate] = 0.0;
            for(int kstate=0; kstate<nstate; kstate++){
                dissipation_matrix[istate][jstate] += eigenvector_matrix[istate][kstate] //R
                                                    * eigenvector_matrix[jstate][kstate] //R^T
                                                    * eigenvalues[kstate]
                                                    * eigenvalue_scale[kstate];
            }
        }
    }

    std::array<real,nstate> entropy_var_int = euler_physics->compute_entropy_variables(soln_int);
    std::array<real,nstate> entropy_var_ext = euler_physics->compute_entropy_variables(soln_ext);

    std::array<real,nstate> dissipation;
    for(int istate=0;istate<nstate; istate++){
        dissipation[istate] = 0.0;
        for(int jstate=0; jstate<nstate; jstate++){
            dissipation[istate] -= 0.5 * dissipation_matrix[istate][jstate]
                              //   * (entropy_var_ext[jstate] - entropy_var_int[jstate])
                              //   * euler_physics->gamm1;
                                 * (entropy_var_ext[jstate] - entropy_var_int[jstate]);
        }
    }

    return dissipation;
    #endif


    #if 0
    // See Blazek 2015, p.103-105
    // -- Note: Modified calculation of alpha_{3,4} to use
    //          dVt (jump in tangential velocities);
    //          expressions are equivalent

    // Blazek 2015
    // p. 103-105
    // Note: This is in fact the Roe-Pike method of Roe & Pike (1984 - Efficient)
    const std::array<real,nstate> prim_soln_int = euler_physics->convert_conservative_to_primitive(soln_int);
    const std::array<real,nstate> prim_soln_ext = euler_physics->convert_conservative_to_primitive(soln_ext);
    // Left cell
    const real density_L = prim_soln_int[0];
    const dealii::Tensor< 1,dim,real > velocities_L = euler_physics->extract_velocities_from_primitive(prim_soln_int);
    const real pressure_L = prim_soln_int[nstate-1];

    //const real normal_vel_L = velocities_L*normal_int;
    real normal_vel_L = 0.0;
    for (int d=0; d<dim; ++d) {
        normal_vel_L+= velocities_L[d]*normal_int[d];
    }
    const real specific_enthalpy_L = euler_physics->compute_specific_enthalpy(soln_int, pressure_L);

    // Right cell
    const real density_R = prim_soln_ext[0];
    const dealii::Tensor< 1,dim,real > velocities_R = euler_physics->extract_velocities_from_primitive(prim_soln_ext);
    const real pressure_R = prim_soln_ext[nstate-1];

    //const real normal_vel_R = velocities_R*normal_int;
    real normal_vel_R = 0.0;
    for (int d=0; d<dim; ++d) {
        normal_vel_R+= velocities_R[d]*normal_int[d];
    }
    const real specific_enthalpy_R = euler_physics->compute_specific_enthalpy(soln_ext, pressure_R);

    // Roe-averaged states
    const real r = sqrt(density_R/density_L);
    const real rp1 = r+1.0;

    const real density_ravg = r*density_L;
    //const dealii::Tensor< 1,dim,real > velocities_ravg = (r*velocities_R + velocities_L) / rp1;
    dealii::Tensor< 1,dim,real > velocities_ravg;
    for (int d=0; d<dim; ++d) {
        velocities_ravg[d] = (r*velocities_R[d] + velocities_L[d]) / rp1;
    }
    const real specific_total_enthalpy_ravg = (r*specific_enthalpy_R + specific_enthalpy_L) / rp1;

    const real vel2_ravg = euler_physics->compute_velocity_squared (velocities_ravg);
    //const real normal_vel_ravg = velocities_ravg*normal_int;
    real normal_vel_ravg = 0.0;
    for (int d=0; d<dim; ++d) {
        normal_vel_ravg += velocities_ravg[d]*normal_int[d];
    }

    const real sound2_ravg = euler_physics->gamm1*(specific_total_enthalpy_ravg-0.5*vel2_ravg);
    real sound_ravg = 1e10;
    if (sound2_ravg > 0.0) {
        sound_ravg = sqrt(sound2_ravg);
    }

    // Compute eigenvalues
    std::array<real, 3> eig_ravg;
    eig_ravg[0] = abs(normal_vel_ravg-sound_ravg);
    eig_ravg[1] = abs(normal_vel_ravg);
    eig_ravg[2] = abs(normal_vel_ravg+sound_ravg);

    const real sound_L = euler_physics->compute_sound(density_L, pressure_L);
    std::array<real, 3> eig_L;
    eig_L[0] = abs(normal_vel_L-sound_L);
    eig_L[1] = abs(normal_vel_L);
    eig_L[2] = abs(normal_vel_L+sound_L);

    const real sound_R = euler_physics->compute_sound(density_R, pressure_R);
    std::array<real, 3> eig_R;
    eig_R[0] = abs(normal_vel_R-sound_R);
    eig_R[1] = abs(normal_vel_R);
    eig_R[2] = abs(normal_vel_R+sound_R);

    // Jumps in pressure and density
    const real dp = pressure_R - pressure_L;
    const real drho = density_R - density_L;

    // Jump in normal velocity
    real dVn = normal_vel_R-normal_vel_L;

    // Jumps in tangential velocities
    dealii::Tensor<1,dim,real> dVt;
    for (int d=0;d<dim;d++) {
        dVt[d] = (velocities_R[d] - velocities_L[d]) - dVn*normal_int[d];
    }

    // Evaluate entropy fix on wave speeds
    evaluate_entropy_fix (eig_L, eig_R, eig_ravg, vel2_ravg, sound_ravg);

    // Evaluate additional modifications to the Roe-Pike scheme (if applicable)
    evaluate_additional_modifications (soln_int, soln_ext, eig_L, eig_R, dVn, dVt);

    // Product of eigenvalues and wave strengths
    real coeff[4];
    coeff[0] = eig_ravg[0]*(dp-density_ravg*sound_ravg*dVn)/(2.0*sound2_ravg);
    coeff[1] = eig_ravg[1]*(drho - dp/sound2_ravg);
    coeff[2] = eig_ravg[1]*density_ravg;
    coeff[3] = eig_ravg[2]*(dp+density_ravg*sound_ravg*dVn)/(2.0*sound2_ravg);

    // Evaluate |A_Roe| * (W_R - W_L)
    std::array<real,nstate> AdW;

    // Vn-c (i=1)
    AdW[0] = coeff[0] * 1.0;
    for (int d=0;d<dim;d++) {
        AdW[1+d] = coeff[0] * (velocities_ravg[d] - sound_ravg * normal_int[d]);
    }
    AdW[nstate-1] = coeff[0] * (specific_total_enthalpy_ravg - sound_ravg*normal_vel_ravg);

    // Vn (i=2)
    AdW[0] += coeff[1] * 1.0;
    for (int d=0;d<dim;d++) {
        AdW[1+d] += coeff[1] * velocities_ravg[d];
    }
    AdW[nstate-1] += coeff[1] * vel2_ravg * 0.5;

    // (i=3,4)
    AdW[0] += coeff[2] * 0.0;
    real dVt_dot_vel_ravg = 0.0;
    for (int d=0;d<dim;d++) {
        AdW[1+d] += coeff[2]*dVt[d];
        dVt_dot_vel_ravg += velocities_ravg[d]*dVt[d];
    }
    AdW[nstate-1] += coeff[2]*dVt_dot_vel_ravg;

    // Vn+c (i=5)
    AdW[0] += coeff[3] * 1.0;
    for (int d=0;d<dim;d++) {
        AdW[1+d] += coeff[3] * (velocities_ravg[d] + sound_ravg * normal_int[d]);
    }
    AdW[nstate-1] += coeff[3] * (specific_total_enthalpy_ravg + sound_ravg*normal_vel_ravg);

    std::array<real, nstate> numerical_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        numerical_flux_dot_n[s] = - 0.5 * AdW[s];
    }

    return numerical_flux_dot_n;
    #endif
#if 0
    // See Blazek 2015, p.103-105
    // -- Note: Modified calculation of alpha_{3,4} to use 
    //          dVt (jump in tangential velocities);
    //          expressions are equivalent
    
    // Blazek 2015
    // p. 103-105
    // Note: This is in fact the Roe-Pike method of Roe & Pike (1984 - Efficient)
    const std::array<real,nstate> prim_soln_int = euler_physics->convert_conservative_to_primitive(soln_int);
    const std::array<real,nstate> prim_soln_ext = euler_physics->convert_conservative_to_primitive(soln_ext);
    // Left cell
    const real density_L = prim_soln_int[0];
    const dealii::Tensor< 1,dim,real > velocities_L = euler_physics->extract_velocities_from_primitive(prim_soln_int);
    const real pressure_L = prim_soln_int[nstate-1];

    //const real normal_vel_L = velocities_L*normal_int;
    real normal_vel_L = 0.0;
    for (int d=0; d<dim; ++d) {
        normal_vel_L+= velocities_L[d]*normal_int[d];
    }
    const real specific_enthalpy_L = euler_physics->compute_specific_enthalpy(soln_int, pressure_L);

    // Right cell
    const real density_R = prim_soln_ext[0];
    const dealii::Tensor< 1,dim,real > velocities_R = euler_physics->extract_velocities_from_primitive(prim_soln_ext);
    const real pressure_R = prim_soln_ext[nstate-1];

    //const real normal_vel_R = velocities_R*normal_int;
    real normal_vel_R = 0.0;
    for (int d=0; d<dim; ++d) {
        normal_vel_R+= velocities_R[d]*normal_int[d];
    }
    const real specific_enthalpy_R = euler_physics->compute_specific_enthalpy(soln_ext, pressure_R);

    // Roe-averaged states
    const real r = sqrt(density_R/density_L);
    const real rp1 = r+1.0;

    const real density_ravg = r*density_L;
    //const dealii::Tensor< 1,dim,real > velocities_ravg = (r*velocities_R + velocities_L) / rp1;
    dealii::Tensor< 1,dim,real > velocities_ravg;
    for (int d=0; d<dim; ++d) {
        velocities_ravg[d] = (r*velocities_R[d] + velocities_L[d]) / rp1;
    }
    const real specific_total_enthalpy_ravg = (r*specific_enthalpy_R + specific_enthalpy_L) / rp1;

    const real vel2_ravg = euler_physics->compute_velocity_squared (velocities_ravg);
    //const real normal_vel_ravg = velocities_ravg*normal_int;
    real normal_vel_ravg = 0.0;
    for (int d=0; d<dim; ++d) {
        normal_vel_ravg += velocities_ravg[d]*normal_int[d];
    }

    const real sound2_ravg = euler_physics->gamm1*(specific_total_enthalpy_ravg-0.5*vel2_ravg);
    real sound_ravg = 1e10;
    if (sound2_ravg > 0.0) {
        sound_ravg = sqrt(sound2_ravg);
    }

    // Compute eigenvalues
    std::array<real, 3> eig_ravg;
    eig_ravg[0] = abs(normal_vel_ravg-sound_ravg);
    eig_ravg[1] = abs(normal_vel_ravg);
    eig_ravg[2] = abs(normal_vel_ravg+sound_ravg);

    const real sound_L = euler_physics->compute_sound(density_L, pressure_L);
    std::array<real, 3> eig_L;
    eig_L[0] = abs(normal_vel_L-sound_L);
    eig_L[1] = abs(normal_vel_L);
    eig_L[2] = abs(normal_vel_L+sound_L);

    const real sound_R = euler_physics->compute_sound(density_R, pressure_R);
    std::array<real, 3> eig_R;
    eig_R[0] = abs(normal_vel_R-sound_R);
    eig_R[1] = abs(normal_vel_R);
    eig_R[2] = abs(normal_vel_R+sound_R);

    // Jumps in pressure and density
    const real dp = pressure_R - pressure_L;
    const real drho = density_R - density_L;

    // Jump in normal velocity
    real dVn = normal_vel_R-normal_vel_L;

    // Jumps in tangential velocities
    dealii::Tensor<1,dim,real> dVt;
    for (int d=0;d<dim;d++) {
        dVt[d] = (velocities_R[d] - velocities_L[d]) - dVn*normal_int[d];
    }

    // Evaluate entropy fix on wave speeds
    evaluate_entropy_fix (eig_L, eig_R, eig_ravg, vel2_ravg, sound_ravg);

    // Evaluate additional modifications to the Roe-Pike scheme (if applicable)
    evaluate_additional_modifications (soln_int, soln_ext, eig_L, eig_R, dVn, dVt);

    // Product of eigenvalues and wave strengths
    real coeff[4];
    coeff[0] = eig_ravg[0]*(dp-density_ravg*sound_ravg*dVn)/(2.0*sound2_ravg);
    coeff[1] = eig_ravg[1]*(drho - dp/sound2_ravg);
    coeff[2] = eig_ravg[1]*density_ravg;
    coeff[3] = eig_ravg[2]*(dp+density_ravg*sound_ravg*dVn)/(2.0*sound2_ravg);

    // Evaluate |A_Roe| * (W_R - W_L)
    std::array<real,nstate> AdW;

    // Vn-c (i=1)
    AdW[0] = coeff[0] * 1.0;
    for (int d=0;d<dim;d++) {
        AdW[1+d] = coeff[0] * (velocities_ravg[d] - sound_ravg * normal_int[d]);
    }
    AdW[nstate-1] = coeff[0] * (specific_total_enthalpy_ravg - sound_ravg*normal_vel_ravg);

    // Vn (i=2)
    AdW[0] += coeff[1] * 1.0;
    for (int d=0;d<dim;d++) {
        AdW[1+d] += coeff[1] * velocities_ravg[d];
    }
    AdW[nstate-1] += coeff[1] * vel2_ravg * 0.5;

    // (i=3,4)
    AdW[0] += coeff[2] * 0.0;
    real dVt_dot_vel_ravg = 0.0;
    for (int d=0;d<dim;d++) {
        AdW[1+d] += coeff[2]*dVt[d];
        dVt_dot_vel_ravg += velocities_ravg[d]*dVt[d];
    }
    AdW[nstate-1] += coeff[2]*dVt_dot_vel_ravg;

    // Vn+c (i=5)
    AdW[0] += coeff[3] * 1.0;
    for (int d=0;d<dim;d++) {
        AdW[1+d] += coeff[3] * (velocities_ravg[d] + sound_ravg * normal_int[d]);
    }
    AdW[nstate-1] += coeff[3] * (specific_total_enthalpy_ravg + sound_ravg*normal_vel_ravg);

    std::array<real, nstate> numerical_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        numerical_flux_dot_n[s] = - 0.5 * AdW[s];
    }

    return numerical_flux_dot_n;
#endif
}

// Instantiation
template class NumericalFluxConvective<PHILIP_DIM, 1, double>;
template class NumericalFluxConvective<PHILIP_DIM, 2, double>;
template class NumericalFluxConvective<PHILIP_DIM, 3, double>;
template class NumericalFluxConvective<PHILIP_DIM, 4, double>;
template class NumericalFluxConvective<PHILIP_DIM, 5, double>;
template class NumericalFluxConvective<PHILIP_DIM, 6, double>;
template class NumericalFluxConvective<PHILIP_DIM, 1, FadType >;
template class NumericalFluxConvective<PHILIP_DIM, 2, FadType >;
template class NumericalFluxConvective<PHILIP_DIM, 3, FadType >;
template class NumericalFluxConvective<PHILIP_DIM, 4, FadType >;
template class NumericalFluxConvective<PHILIP_DIM, 5, FadType >;
template class NumericalFluxConvective<PHILIP_DIM, 6, FadType >;
template class NumericalFluxConvective<PHILIP_DIM, 1, RadType >;
template class NumericalFluxConvective<PHILIP_DIM, 2, RadType >;
template class NumericalFluxConvective<PHILIP_DIM, 3, RadType >;
template class NumericalFluxConvective<PHILIP_DIM, 4, RadType >;
template class NumericalFluxConvective<PHILIP_DIM, 5, RadType >;
template class NumericalFluxConvective<PHILIP_DIM, 6, RadType >;

template class NumericalFluxConvective<PHILIP_DIM, 1, FadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 2, FadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 3, FadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 4, FadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 5, FadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 6, FadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 1, RadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 2, RadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 3, RadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 4, RadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 5, RadFadType >;
template class NumericalFluxConvective<PHILIP_DIM, 6, RadFadType >;

template class LaxFriedrichs<PHILIP_DIM, 1, double>;
template class LaxFriedrichs<PHILIP_DIM, 2, double>;
template class LaxFriedrichs<PHILIP_DIM, 3, double>;
template class LaxFriedrichs<PHILIP_DIM, 4, double>;
template class LaxFriedrichs<PHILIP_DIM, 5, double>;
template class LaxFriedrichs<PHILIP_DIM, 6, double>;
template class LaxFriedrichs<PHILIP_DIM, 1, FadType >;
template class LaxFriedrichs<PHILIP_DIM, 2, FadType >;
template class LaxFriedrichs<PHILIP_DIM, 3, FadType >;
template class LaxFriedrichs<PHILIP_DIM, 4, FadType >;
template class LaxFriedrichs<PHILIP_DIM, 5, FadType >;
template class LaxFriedrichs<PHILIP_DIM, 6, FadType >;
template class LaxFriedrichs<PHILIP_DIM, 1, RadType >;
template class LaxFriedrichs<PHILIP_DIM, 2, RadType >;
template class LaxFriedrichs<PHILIP_DIM, 3, RadType >;
template class LaxFriedrichs<PHILIP_DIM, 4, RadType >;
template class LaxFriedrichs<PHILIP_DIM, 5, RadType >;
template class LaxFriedrichs<PHILIP_DIM, 6, RadType >;
template class LaxFriedrichs<PHILIP_DIM, 1, FadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 2, FadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 3, FadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 4, FadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 5, FadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 6, FadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 1, RadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 2, RadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 3, RadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 4, RadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 5, RadFadType >;
template class LaxFriedrichs<PHILIP_DIM, 6, RadFadType >;

template class RoePike<PHILIP_DIM, PHILIP_DIM+2, double>;
template class RoePike<PHILIP_DIM, PHILIP_DIM+2, FadType >;
template class RoePike<PHILIP_DIM, PHILIP_DIM+2, RadType >;
template class RoePike<PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class RoePike<PHILIP_DIM, PHILIP_DIM+2, RadFadType >;
template class L2Roe<PHILIP_DIM, PHILIP_DIM+2, double>;
template class L2Roe<PHILIP_DIM, PHILIP_DIM+2, FadType >;
template class L2Roe<PHILIP_DIM, PHILIP_DIM+2, RadType >;
template class L2Roe<PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class L2Roe<PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

template class Central<PHILIP_DIM, 1, double>;
template class Central<PHILIP_DIM, 2, double>;
template class Central<PHILIP_DIM, 3, double>;
template class Central<PHILIP_DIM, 4, double>;
template class Central<PHILIP_DIM, 5, double>;
//template class Central<PHILIP_DIM, 6, double>;
template class Central<PHILIP_DIM, 1, FadType >;
template class Central<PHILIP_DIM, 2, FadType >;
template class Central<PHILIP_DIM, 3, FadType >;
template class Central<PHILIP_DIM, 4, FadType >;
template class Central<PHILIP_DIM, 5, FadType >;
//template class Central<PHILIP_DIM, 6, FadType >;
template class Central<PHILIP_DIM, 1, RadType >;
template class Central<PHILIP_DIM, 2, RadType >;
template class Central<PHILIP_DIM, 3, RadType >;
template class Central<PHILIP_DIM, 4, RadType >;
template class Central<PHILIP_DIM, 5, RadType >;
//template class Central<PHILIP_DIM, 6, RadType >;
template class Central<PHILIP_DIM, 1, FadFadType >;
template class Central<PHILIP_DIM, 2, FadFadType >;
template class Central<PHILIP_DIM, 3, FadFadType >;
template class Central<PHILIP_DIM, 4, FadFadType >;
template class Central<PHILIP_DIM, 5, FadFadType >;
//template class Central<PHILIP_DIM, 6, FadFadType >;
template class Central<PHILIP_DIM, 1, RadFadType >;
template class Central<PHILIP_DIM, 2, RadFadType >;
template class Central<PHILIP_DIM, 3, RadFadType >;
template class Central<PHILIP_DIM, 4, RadFadType >;
template class Central<PHILIP_DIM, 5, RadFadType >;
//template class Central<PHILIP_DIM, 6, RadFadType >;

template class EntropyConserving<PHILIP_DIM, 1, double>;
template class EntropyConserving<PHILIP_DIM, 2, double>;
template class EntropyConserving<PHILIP_DIM, 3, double>;
template class EntropyConserving<PHILIP_DIM, 4, double>;
template class EntropyConserving<PHILIP_DIM, 5, double>;
//template class EntropyConserving<PHILIP_DIM, 6, double>;
template class EntropyConserving<PHILIP_DIM, 1, FadType >;
template class EntropyConserving<PHILIP_DIM, 2, FadType >;
template class EntropyConserving<PHILIP_DIM, 3, FadType >;
template class EntropyConserving<PHILIP_DIM, 4, FadType >;
template class EntropyConserving<PHILIP_DIM, 5, FadType >;
//template class EntropyConserving<PHILIP_DIM, 6, FadType >;
template class EntropyConserving<PHILIP_DIM, 1, RadType >;
template class EntropyConserving<PHILIP_DIM, 2, RadType >;
template class EntropyConserving<PHILIP_DIM, 3, RadType >;
template class EntropyConserving<PHILIP_DIM, 4, RadType >;
template class EntropyConserving<PHILIP_DIM, 5, RadType >;
//template class EntropyConserving<PHILIP_DIM, 6, RadType >;
template class EntropyConserving<PHILIP_DIM, 1, FadFadType >;
template class EntropyConserving<PHILIP_DIM, 2, FadFadType >;
template class EntropyConserving<PHILIP_DIM, 3, FadFadType >;
template class EntropyConserving<PHILIP_DIM, 4, FadFadType >;
template class EntropyConserving<PHILIP_DIM, 5, FadFadType >;
//template class EntropyConserving<PHILIP_DIM, 6, FadFadType >;
template class EntropyConserving<PHILIP_DIM, 1, RadFadType >;
template class EntropyConserving<PHILIP_DIM, 2, RadFadType >;
template class EntropyConserving<PHILIP_DIM, 3, RadFadType >;
template class EntropyConserving<PHILIP_DIM, 4, RadFadType >;
template class EntropyConserving<PHILIP_DIM, 5, RadFadType >;
//template class EntropyConserving<PHILIP_DIM, 6, RadFadType >;

template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 1, double>;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 2, double>;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 3, double>;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 4, double>;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 5, double>;
//template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 6, double>;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 1, FadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 2, FadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 3, FadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 4, FadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 5, FadType >;
//template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 6, FadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 1, RadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 2, RadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 3, RadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 4, RadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 5, RadType >;
//template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 6, RadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 1, FadFadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 2, FadFadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 3, FadFadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 4, FadFadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 5, FadFadType >;
//template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 6, FadFadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 1, RadFadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 2, RadFadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 3, RadFadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 4, RadFadType >;
template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 5, RadFadType >;
//template class EntropyConservingWithLaxFriedrichsDissipation<PHILIP_DIM, 6, RadFadType >;

template class EntropyConservingWithRoeDissipation<PHILIP_DIM, PHILIP_DIM+2, double>;
template class EntropyConservingWithRoeDissipation<PHILIP_DIM, PHILIP_DIM+2, FadType >;
template class EntropyConservingWithRoeDissipation<PHILIP_DIM, PHILIP_DIM+2, RadType >;
template class EntropyConservingWithRoeDissipation<PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class EntropyConservingWithRoeDissipation<PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

template class EntropyConservingWithL2RoeDissipation<PHILIP_DIM, PHILIP_DIM+2, double>;
template class EntropyConservingWithL2RoeDissipation<PHILIP_DIM, PHILIP_DIM+2, FadType >;
template class EntropyConservingWithL2RoeDissipation<PHILIP_DIM, PHILIP_DIM+2, RadType >;
template class EntropyConservingWithL2RoeDissipation<PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class EntropyConservingWithL2RoeDissipation<PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

//template class EntropyConservingWithRoeDissipation<PHILIP_DIM, PHILIP_DIM+3, double>;
//template class EntropyConservingWithRoeDissipation<PHILIP_DIM, PHILIP_DIM+3, FadType >;
//template class EntropyConservingWithRoeDissipation<PHILIP_DIM, PHILIP_DIM+3, RadType >;
//template class EntropyConservingWithRoeDissipation<PHILIP_DIM, PHILIP_DIM+3, FadFadType >;
//template class EntropyConservingWithRoeDissipation<PHILIP_DIM, PHILIP_DIM+3, RadFadType >;
//
//template class EntropyConservingWithL2RoeDissipation<PHILIP_DIM, PHILIP_DIM+3, double>;
//template class EntropyConservingWithL2RoeDissipation<PHILIP_DIM, PHILIP_DIM+3, FadType >;
//template class EntropyConservingWithL2RoeDissipation<PHILIP_DIM, PHILIP_DIM+3, RadType >;
//template class EntropyConservingWithL2RoeDissipation<PHILIP_DIM, PHILIP_DIM+3, FadFadType >;
//template class EntropyConservingWithL2RoeDissipation<PHILIP_DIM, PHILIP_DIM+3, RadFadType >;

template class BaselineNumericalFluxConvective<PHILIP_DIM, 1, double>;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 2, double>;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 3, double>;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 4, double>;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 5, double>;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 6, double>;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 1, FadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 2, FadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 3, FadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 4, FadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 5, FadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 6, FadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 1, RadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 2, RadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 3, RadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 4, RadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 5, RadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 6, RadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 1, FadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 2, FadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 3, FadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 4, FadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 5, FadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 6, FadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 1, RadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 2, RadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 3, RadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 4, RadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 5, RadFadType >;
template class BaselineNumericalFluxConvective<PHILIP_DIM, 6, RadFadType >;

template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 1, double>;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 2, double>;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 3, double>;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 4, double>;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 5, double>;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 6, double>;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 1, FadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 2, FadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 3, FadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 4, FadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 5, FadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 6, FadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 1, RadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 2, RadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 3, RadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 4, RadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 5, RadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 6, RadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 1, FadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 2, FadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 3, FadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 4, FadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 5, FadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 6, FadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 1, RadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 2, RadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 3, RadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 4, RadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 5, RadFadType >;
template class CentralBaselineNumericalFluxConvective<PHILIP_DIM, 6, RadFadType >;

template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 1, double>;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 2, double>;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 3, double>;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 4, double>;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 5, double>;
//template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 6, double>;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 1, FadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 2, FadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 3, FadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 4, FadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 5, FadType >;
//template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 6, FadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 1, RadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 2, RadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 3, RadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 4, RadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 5, RadType >;
//template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 6, RadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 1, FadFadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 2, FadFadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 3, FadFadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 4, FadFadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 5, FadFadType >;
//template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 6, FadFadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 1, RadFadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 2, RadFadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 3, RadFadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 4, RadFadType >;
template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 5, RadFadType >;
//template class EntropyConservingBaselineNumericalFluxConvective<PHILIP_DIM, 6, RadFadType >;

template class RiemannSolverDissipation<PHILIP_DIM, 1, double>;
template class RiemannSolverDissipation<PHILIP_DIM, 2, double>;
template class RiemannSolverDissipation<PHILIP_DIM, 3, double>;
template class RiemannSolverDissipation<PHILIP_DIM, 4, double>;
template class RiemannSolverDissipation<PHILIP_DIM, 5, double>;
template class RiemannSolverDissipation<PHILIP_DIM, 6, double>;
template class RiemannSolverDissipation<PHILIP_DIM, 1, FadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 2, FadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 3, FadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 4, FadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 5, FadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 6, FadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 1, RadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 2, RadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 3, RadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 4, RadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 5, RadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 6, RadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 1, FadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 2, FadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 3, FadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 4, FadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 5, FadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 6, FadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 1, RadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 2, RadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 3, RadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 4, RadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 5, RadFadType >;
template class RiemannSolverDissipation<PHILIP_DIM, 6, RadFadType >;

template class ZeroRiemannSolverDissipation<PHILIP_DIM, 1, double>;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 2, double>;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 3, double>;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 4, double>;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 5, double>;
//template class ZeroRiemannSolverDissipation<PHILIP_DIM, 6, double>;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 1, FadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 2, FadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 3, FadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 4, FadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 5, FadType >;
//template class ZeroRiemannSolverDissipation<PHILIP_DIM, 6, FadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 1, RadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 2, RadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 3, RadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 4, RadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 5, RadType >;
//template class ZeroRiemannSolverDissipation<PHILIP_DIM, 6, RadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 1, FadFadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 2, FadFadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 3, FadFadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 4, FadFadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 5, FadFadType >;
//template class ZeroRiemannSolverDissipation<PHILIP_DIM, 6, FadFadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 1, RadFadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 2, RadFadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 3, RadFadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 4, RadFadType >;
template class ZeroRiemannSolverDissipation<PHILIP_DIM, 5, RadFadType >;
//template class ZeroRiemannSolverDissipation<PHILIP_DIM, 6, RadFadType >;

template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 1, double>;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 2, double>;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 3, double>;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 4, double>;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 5, double>;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 6, double>;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 1, FadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 2, FadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 3, FadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 4, FadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 5, FadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 6, FadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 1, RadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 2, RadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 3, RadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 4, RadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 5, RadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 6, RadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 1, FadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 2, FadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 3, FadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 4, FadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 5, FadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 6, FadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 1, RadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 2, RadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 3, RadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 4, RadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 5, RadFadType >;
template class LaxFriedrichsRiemannSolverDissipation<PHILIP_DIM, 6, RadFadType >;

template class RoeBaseRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, double>;
template class RoeBaseRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, FadType >;
template class RoeBaseRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, RadType >;
template class RoeBaseRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class RoeBaseRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

template class RoePikeRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, double>;
template class RoePikeRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, FadType >;
template class RoePikeRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, RadType >;
template class RoePikeRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class RoePikeRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

template class L2RoeRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, double>;
template class L2RoeRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, FadType >;
template class L2RoeRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, RadType >;
template class L2RoeRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class L2RoeRiemannSolverDissipation<PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

} // NumericalFlux namespace
} // PHiLiP namespace
