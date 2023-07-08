#include <deal.II/base/function.h>
#include "exact_solution.h"

namespace PHiLiP {

// ========================================================
// ZERO -- Returns zero everywhere; used a placeholder when no exact solution is defined.
// ========================================================
template <int dim, int nstate, typename real>
ExactSolutionFunction_Zero<dim,nstate,real>
::ExactSolutionFunction_Zero(double time_compare)
        : ExactSolutionFunction<dim,nstate,real>()
        , t(time_compare)
{
}

template <int dim, int nstate, typename real>
inline real ExactSolutionFunction_Zero<dim,nstate,real>
::value(const dealii::Point<dim,real> &/*point*/, const unsigned int /*istate*/) const
{
    real value = 0;
    return value;
}

// ========================================================
// 1D SINE -- Exact solution for advection_explicit_time_study
// ========================================================
template <int dim, int nstate, typename real>
ExactSolutionFunction_1DSine<dim,nstate,real>
::ExactSolutionFunction_1DSine (double time_compare)
        : ExactSolutionFunction<dim,nstate,real>()
        , t(time_compare)
{
}

template <int dim, int nstate, typename real>
inline real ExactSolutionFunction_1DSine<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int /*istate*/) const
{
    double x_adv_speed = 1.0;

    real value = 0;
    real pi = dealii::numbers::PI;
    if(point[0] >= 0.0 && point[0] <= 2.0){
        value = sin(2*pi*(point[0] - x_adv_speed * t)/2.0);
    }
    return value;
}

// ========================================================
// Inviscid Isentropic Vortex 
// ========================================================
template <int dim, int nstate, typename real>
ExactSolutionFunction_IsentropicVortex<dim,nstate,real>
::ExactSolutionFunction_IsentropicVortex(double time_compare)
        : ExactSolutionFunction<dim,nstate,real>()
        , t(time_compare)
{
}

template <int dim, int nstate, typename real>
inline real ExactSolutionFunction_IsentropicVortex<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
//#if 0
    // Setting constants
   // const double L = 10.0; // half-width of domain
    const double L = 5.0; // half-width of domain
    const double pi = dealii::numbers::PI;
    const double gam = 1.4;
    const double M_infty = sqrt(2/gam);
    const double R = 1;
    const double sigma = 1;
    const double beta = M_infty * 5 * sqrt(2.0)/4.0/pi * exp(1.0/2.0);
    const double alpha = pi/4; //rad

    // Centre of the vortex  at t
    const double x_travel = M_infty * t * cos(alpha);
   // const double x0 = 0.0 + x_travel;
    const double x0 = 5.0 + x_travel;
    const double y_travel = M_infty * t * sin(alpha);
   // const double y0 = 0.0 + y_travel;
    const double y0 = 5.0 + y_travel;
    const double x = std::fmod(point[0] - x0-L, 2*L)+L;
    const double y = std::fmod(point[1] - y0-L, 2*L)+L;

    const double Omega = beta * exp(-0.5/sigma/sigma* (x/R * x/R + y/R * y/R));
    const double delta_Ux = -y/R * Omega;
    const double delta_Uy =  x/R * Omega;
    const double delta_T  = -(gam-1.0)/2.0 * Omega * Omega;

    // Primitive
    const double rho = pow((1 + delta_T), 1.0/(gam-1.0));
    const double Ux = M_infty * cos(alpha) + delta_Ux;
    const double Uy = M_infty * sin(alpha) + delta_Uy;
    const double Uz = 0;
    const double p = 1.0/gam*pow(1+delta_T, gam/(gam-1.0));

    //Convert to conservative variables
    if (istate == 0)      return rho;       //density 
    else if (istate == nstate-1) return p/(gam-1.0) + 0.5 * rho * (Ux*Ux + Uy*Uy + Uz*Uz);   //total energy
    else if (istate == 1) return rho * Ux;  //x-momentum
    else if (istate == 2) return rho * Uy;  //y-momentum
    else if (istate == 3) return rho * Uz;  //z-momentum
    else return 0;
//#endif

#if 0
    //Jesse Chan isentropic vortex
    const double Pi_max = 0.4;
    const double c_1 = 5.0;
    const double c_2 = 5.0;
 //   const double c_1 = 0.0;
 //   const double c_2 = 0.0;
  //  const double c_2 = -2.5;
    const double gamma = 1.4;
    const double P_0 = 1.0/gamma;
    const real u0 = 1.0;
    const real v0 = 1.0;

//    const double pi = dealii::numbers::PI;
   // const double length = 4.0 * pi;
//    const double length = 20.0;
    double distance_travelled = t;//v_0 * t with v_0 = t
//    if(distance_travelled > length - c_2)//reached the edge first
//        distance_travelled -= (length - c_2);
//    double distance_in_domain_after_periodicity = fmod(distance_travelled / length, 1.0) * length;
    double distance_in_domain_after_periodicity = distance_travelled;
    //location
    const double x = point[0];
    const double y = point[1];
   // const double r_square = (y - c_2 - t)*(y - c_2 - t) + (x - c_1)*(x - c_1);
    const double r_square = (y - c_2 - v0 * distance_in_domain_after_periodicity)*(y - c_2 - v0 * distance_in_domain_after_periodicity) + (x - c_1 - u0 * distance_in_domain_after_periodicity)*(x - c_1 - u0 * distance_in_domain_after_periodicity);
   // const double r_square = (y - c_2 - t)*(y - c_2 - t) + (x - c_1 - t)*(x - c_1 - t);
    const double Pi = Pi_max * exp(0.5 * (1.0 - r_square));

    //conservative variables
    const real density = pow(1.0 - 0.4 / 2.0 * Pi * Pi, 1.0 / (0.4) );
    const real u = u0 + Pi * ( - (y - c_2 - v0 * distance_in_domain_after_periodicity));
  //  const real u = Pi * ( - (y - c_2 - t));
   // const real v = Pi * ( (x - c_1));
    const real v = v0 + Pi * ( (x - c_1 - u0 * distance_in_domain_after_periodicity));
   // const real v = Pi * ( (x - c_1 - t));
    const real pressure = P_0 * pow(density, gamma);
    // Primitive
    std::array<real,nstate> soln_conservative;
    soln_conservative[0] = density;
    soln_conservative[1] = density * u;
    soln_conservative[2] = density * v;
    #if PHILIP_DIM==3
    soln_conservative[3] = 0.0;
    #endif
    soln_conservative[nstate-1] = pressure / 0.4 + 0.5 * density * (u*u + v*v);
    return soln_conservative[istate];
//    if(istate == 0){
//        return density;
//    }
//    if(istate == 1){
//        return density * u;
//    }
//    if(istate == 2){
//        return density * v;
//    }
//    if(istate == 3 && dim == 2){
//        const real rho_e = P_0 / 0.4 * pow(density,gamma) +  density / 2.0 * ( u * u + v * v);
//        return rho_e;
//    }
//    if(istate == 3 && dim == 3){
//        return 0.0;
//    }
//    if(istate == 4){
//        const real rho_e = P_0 / 0.4 * pow(density,gamma) +  density / 2.0 * ( u * u + v * v);
//        return rho_e;
//    }
//    else return 0;
#endif

}

//=========================================================
// FLOW SOLVER -- Exact Solution Base Class + Factory
//=========================================================
template <int dim, int nstate, typename real>
ExactSolutionFunction<dim,nstate,real>
::ExactSolutionFunction ()
    : dealii::Function<dim,real>(nstate)
{
    //do nothing
}

template <int dim, int nstate, typename real>
std::shared_ptr<ExactSolutionFunction<dim, nstate, real>>
ExactSolutionFactory<dim,nstate, real>::create_ExactSolutionFunction(
        const Parameters::FlowSolverParam& flow_solver_parameters, 
        const double time_compare)
{
    // Get the flow case type
    const FlowCaseEnum flow_type = flow_solver_parameters.flow_case_type;
    if (flow_type == FlowCaseEnum::periodic_1D_unsteady){
        if constexpr (dim==1 && nstate==dim)  return std::make_shared<ExactSolutionFunction_1DSine<dim,nstate,real> > (time_compare);
    } else if (flow_type == FlowCaseEnum::isentropic_vortex){
        if constexpr (dim>1 && nstate==dim+2)  return std::make_shared<ExactSolutionFunction_IsentropicVortex<dim,nstate,real> > (time_compare);
    } else {
        // Select zero function if there is no exact solution defined
        return std::make_shared<ExactSolutionFunction_Zero<dim,nstate,real>> (time_compare);
    }
    return nullptr;
}

template class ExactSolutionFunction <PHILIP_DIM,PHILIP_DIM, double>;
template class ExactSolutionFunction <PHILIP_DIM,PHILIP_DIM+2, double>;
template class ExactSolutionFactory <PHILIP_DIM, PHILIP_DIM+2, double>;
template class ExactSolutionFactory <PHILIP_DIM, PHILIP_DIM, double>;
template class ExactSolutionFunction_Zero <PHILIP_DIM,1, double>;
template class ExactSolutionFunction_Zero <PHILIP_DIM,2, double>;
template class ExactSolutionFunction_Zero <PHILIP_DIM,3, double>;
template class ExactSolutionFunction_Zero <PHILIP_DIM,4, double>;
template class ExactSolutionFunction_Zero <PHILIP_DIM,5, double>;

#if PHILIP_DIM>1
template class ExactSolutionFunction_IsentropicVortex <PHILIP_DIM,PHILIP_DIM+2, double>;
#endif
} // PHiLiP namespace
