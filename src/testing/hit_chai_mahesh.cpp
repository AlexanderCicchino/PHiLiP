#include <fstream>
#include "dg/dg_factory.hpp"
#include "hit_chai_mahesh.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "physics/initial_conditions/initial_condition_function.h"
#include "mesh/grids/straight_periodic_cube.hpp"

#include <eigen/Eigen/Eigenvalues>
#include <eigen/Eigen/Dense>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
HITChaiMahesh<dim, nstate>::HITChaiMahesh(const Parameters::AllParameters *const parameters_input)
    : TestsBase::TestsBase(parameters_input)
{}

//template<int dim, int nstate>
//double HITChaiMahesh<dim, nstate>::set_initial_condition_from_file(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree, const double delta_x) const
//{
//
//}
template<int dim, int nstate>
double HITChaiMahesh<dim, nstate>::get_timestep(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree, const double delta_x) const
{
    //get local CFL
   // const unsigned int n_dofs_cell = poly_degree==0 ? 6 : nstate*pow(poly_degree+1,dim);
    const unsigned int n_dofs_cell = poly_degree==0 ? 16 : nstate*pow(poly_degree+1,dim);
    const unsigned int n_quad_pts = pow(poly_degree+1,dim);
    std::vector<dealii::types::global_dof_index> dofs_indices1 (n_dofs_cell);

    double cfl_min = 1e100;
    std::shared_ptr < Physics::PhysicsBase<dim, nstate, double > > pde_physics_double  = PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters);
    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;

        cell->get_dof_indices (dofs_indices1);
        std::vector< std::array<double,nstate>> soln_at_q(n_quad_pts);
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            for (int istate=0; istate<nstate; istate++) {
                soln_at_q[iquad][istate]      = 0;
            }
        }
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
          dealii::Point<dim> qpoint = dg->volume_quadrature_collection[poly_degree].point(iquad);
            for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
                soln_at_q[iquad][istate] += dg->solution[dofs_indices1[idof]] * dg->fe_collection[poly_degree].shape_value_component(idof, qpoint, istate);
            }
        }

        std::vector< double > convective_eigenvalues(n_quad_pts);
        for (unsigned int isol = 0; isol < n_quad_pts; ++isol) {
            convective_eigenvalues[isol] = pde_physics_double->max_convective_eigenvalue (soln_at_q[isol]);
        }
        const double max_eig = *(std::max_element(convective_eigenvalues.begin(), convective_eigenvalues.end()));
        double cfl = delta_x/max_eig * dg->all_parameters->flow_solver_param.courant_friedrichs_lewy_number;

        if(cfl < cfl_min)
            cfl_min = cfl;
    }
    return cfl_min;
}

template <int dim, int nstate>
int HITChaiMahesh<dim, nstate>::run_test() const
{
    using real = double;

    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;  
    const double left = all_parameters_new.flow_solver_param.grid_left_bound;
    const double right = all_parameters_new.flow_solver_param.grid_right_bound;
    const unsigned int grid_degree = all_parameters_new.flow_solver_param.grid_degree;
    const unsigned int num_grid_elem_per_dir = all_parameters_new.flow_solver_param.number_of_grid_elements_per_dimension;
    const unsigned int poly_degree = all_parameters_new.flow_solver_param.poly_degree;

    //Initialize grid
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
        using Triangulation = dealii::Triangulation<dim>;
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
        using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
            MPI_COMM_WORLD,
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));
#endif
    PHiLiP::Grids::straight_periodic_cube<dim,Triangulation>(grid, left, right, num_grid_elem_per_dir );

    // Create DG
    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
    dg->allocate_system ();

    std::cout << "Implement initial conditions" << std::endl;
    //Read in from files, and project


//    set_initial_condition_from_file(dg, poly_degree, delta_x);

    std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function = 
                InitialConditionFactory<dim,nstate,double>::create_InitialConditionFunction(&all_parameters_new);
    SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);


//    SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);


    const unsigned int n_global_active_cells2 = grid->n_global_active_cells();
    double delta_x = (right-left)/pow(n_global_active_cells2,1.0/dim)/(poly_degree+1.0);
    pcout<<" delta x "<<delta_x<<std::endl;

    all_parameters_new.ode_solver_param.initial_time_step =  get_timestep(dg,poly_degree,delta_x);
     
    std::cout << "creating ODE solver" << std::endl;
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    std::cout << "ODE solver successfully created" << std::endl;
    double finalTime = all_parameters_new.flow_solver_param.final_time;
    std::cout<<"Final time "<<finalTime<<std::endl;
    double dt = all_parameters_new.ode_solver_param.initial_time_step;

    ode_solver->current_iteration = 0;
    ode_solver->allocate_ode_system();
    const int file_number = ode_solver->current_iteration / all_parameters_new.ode_solver_param.output_solution_every_x_steps;
    dg->output_results_vtk(file_number);

    while(ode_solver->current_time < finalTime){
        
        dt = get_timestep(dg,poly_degree, delta_x);
        ode_solver->step_in_time(dt,false);
        const bool is_output_iteration = (ode_solver->current_iteration % all_parameters_new.ode_solver_param.output_solution_every_x_steps == 0);
        if (is_output_iteration) {
            const int file_number = ode_solver->current_iteration / all_parameters_new.ode_solver_param.output_solution_every_x_steps;
            dg->output_results_vtk(file_number);
        }
    }

    std::cout<<"Done time integration"<<std::endl;

    return 0;
}

template class HITChaiMahesh <PHILIP_DIM,PHILIP_DIM+2>;

} // Tests namespace
} // PHiLiP namespace



