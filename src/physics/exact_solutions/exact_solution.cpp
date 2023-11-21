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
::ExactSolutionFunction_IsentropicVortex(double time_compare,
        const Parameters::FlowSolverParam& flow_solver_parameters)
        : ExactSolutionFunction<dim,nstate,real>()
        , t(time_compare)
        , x_center((flow_solver_parameters.grid_right_bound + flow_solver_parameters.grid_left_bound)/2.0)
        , y_center((flow_solver_parameters.grid_right_bound + flow_solver_parameters.grid_left_bound)/2.0)
        , length(flow_solver_parameters.grid_right_bound - flow_solver_parameters.grid_left_bound)
        , vortex_strength(flow_solver_parameters.isentropic_vortex_strength)
        , u0(flow_solver_parameters.isentropic_vortex_vel_x)
        , v0(flow_solver_parameters.isentropic_vortex_vel_y)
{
}

template <int dim, int nstate, typename real>
inline real ExactSolutionFunction_IsentropicVortex<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    const double gamma = 1.4;
    const double Pi_max = vortex_strength;
    const double c_1 = x_center;
    const double c_2 = y_center;
    const double P_0 = 1.0/gamma;
    const double rho_0 = 1.0;
    const double radius = 1.0;
  //  const double radius = 0.5;

    double distance_y = fmod((v0*t+c_2)/ length, 1.0) * length - c_2;
    double distance_x = fmod((u0*t+c_1)/ length, 1.0) * length - c_1;

    //location
    const double x = point[0];
    const double y = point[1];
    const double r_square = (y - c_2 - distance_y)*(y - c_2 - distance_y) + (x - c_1 - distance_x)*(x - c_1 - distance_x);
    const double Pi = Pi_max * exp(0.5 * (1.0 - r_square/(radius*radius)));

    //conservative variables
    const real density = pow(1.0 - (gamma-1.0) / 2.0 * Pi * Pi, 1.0 / (gamma-1.0) );
    const real u = u0 + sqrt(gamma*P_0/rho_0) * Pi * ( - (y - c_2 - distance_y));
    const real v = v0 + sqrt(gamma*P_0/rho_0) * Pi * ( (x - c_1 - distance_x));
    const real pressure = P_0 * pow(density, gamma);
    // Primitive
    std::array<real,nstate> soln_conservative;
    soln_conservative[0] = density;
    soln_conservative[1] = density * u;
    soln_conservative[2] = density * v;
    #if PHILIP_DIM==3
    soln_conservative[3] = 0.0;
    #endif
    soln_conservative[nstate-1] = pressure / (gamma-1.0) + 0.5 * density * (u*u + v*v);
    return soln_conservative[istate];
//#endif

}

// ========================================================
// Euler Sine Wave 
// ========================================================
template <int dim, int nstate, typename real>
ExactSolutionFunction_EulerSineWave<dim,nstate,real>
::ExactSolutionFunction_EulerSineWave(double time_compare,
        const Parameters::FlowSolverParam& /*flow_solver_parameters*/)
        : ExactSolutionFunction<dim,nstate,real>()
        , t(time_compare)
{
}

template <int dim, int nstate, typename real>
inline real ExactSolutionFunction_EulerSineWave<dim,nstate,real>
::value(const dealii::Point<dim,real> &point, const unsigned int istate) const
{
    real pi = dealii::numbers::PI;
    
    real density;
    if constexpr(dim == 2){
        density = 2.0 + 1.0/10.0*sin(pi*(point[0] + point[1] - 2.0 * t));
    }
    else{
        density = 2.0 + 1.0/10.0*sin(pi*(point[0] + point[1] + point[2] - 2.0 * t));
    }

    if(istate == (nstate - 1)){
        return density * density;
    }
    else{
        return density;
    }
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
        if constexpr (dim>1 && nstate==dim+2)  return std::make_shared<ExactSolutionFunction_IsentropicVortex<dim,nstate,real> > (time_compare, flow_solver_parameters);
    } else if (flow_type == FlowCaseEnum::euler_sine_wave){
        if constexpr (dim>1 && nstate==dim+2)  return std::make_shared<ExactSolutionFunction_EulerSineWave<dim,nstate,real> > (time_compare, flow_solver_parameters);
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
template class ExactSolutionFunction_EulerSineWave <PHILIP_DIM,PHILIP_DIM+2, double>;
#endif
} // PHiLiP namespace
