#include <fstream>
#include "dg/dg_factory.hpp"
#include "inviscid_vortex_shock_interaction.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "physics/initial_conditions/initial_condition_function.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
InviscidVortexShockInteraction<dim, nstate>::InviscidVortexShockInteraction(const Parameters::AllParameters *const parameters_input)
    : TestsBase::TestsBase(parameters_input)
{}
template<int dim, int nstate>
double InviscidVortexShockInteraction<dim, nstate>::get_timestep(const std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree, const double delta_x) const
{
    //get local CFL
    const unsigned int n_dofs_cell = nstate*pow(poly_degree+1,dim);
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

        double cfl = dg->all_parameters->flow_solver_param.courant_friedrichs_lewy_number
                   * delta_x/max_eig;
        if(cfl < cfl_min)
            cfl_min = cfl;

    }
    return cfl_min;
}

template <int dim, int nstate>
int InviscidVortexShockInteraction<dim, nstate>::run_test() const
{
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    using real = double;

    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;  
    dealii::Point<dim> left;
    dealii::Point<dim> right;
    left[0] = 0.0;
    left[1] = 0.0;
    right[0] = 2.0;
    right[1] = 1.0;
    const int n_cells_y = all_parameters->flow_solver_param.number_of_grid_elements_per_dimension;
    const int n_cells_x = 2 * n_cells_y;
    std::vector<unsigned int> repititions(dim);
    repititions[0] = n_cells_x;
    repititions[1] = n_cells_y;
    unsigned int poly_degree = all_parameters->flow_solver_param.poly_degree;
    unsigned int grid_degree = all_parameters->flow_solver_param.grid_degree;

    //straight grid setup
    dealii::GridGenerator::subdivided_hyper_rectangle(*grid, repititions, left, right, true);
    //apply boundary conditions
    // Set boundary type and design type
    for (typename dealii::parallel::distributed::Triangulation<2>::active_cell_iterator cell = grid->begin_active(); cell != grid->end(); ++cell) {
        for (unsigned int face=0; face<dealii::GeometryInfo<2>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                //x : 0,1, y : 2,3
                if (current_id == 2 || current_id == 3) {
                    cell->face(face)->set_boundary_id (1001); // wall BC
                } 
                if(current_id == 0) {
                    cell->face(face)->set_boundary_id (1003); // supersonic inlet
                }
                if(current_id == 1) {
                    cell->face(face)->set_boundary_id (1004); // Riemann invariance subsonic outlet
                }
            }
        }
    }


    // Create DG
    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
    dg->allocate_system ();

    pcout << "Implement initial conditions" << std::endl;
    std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function = 
                InitialConditionFactory<dim,nstate,double>::create_InitialConditionFunction(&all_parameters_new);
    SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);

    double delta_x = (right[1]-left[1])/n_cells_y/(poly_degree+1.0);
    pcout<<" delta x "<<delta_x<<std::endl;

    all_parameters_new.ode_solver_param.initial_time_step =  get_timestep(dg,poly_degree,delta_x);
     
    pcout << "creating ODE solver" << std::endl;
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    pcout << "ODE solver successfully created" << std::endl;
    double finalTime = all_parameters->flow_solver_param.final_time;

    pcout << " number dofs " << dg->dof_handler.n_dofs()<<std::endl;
    pcout << "preparing to advance solution in time" << std::endl;

    ode_solver->current_iteration = 0;
    ode_solver->allocate_ode_system();

    //output IC
    const int file_number = ode_solver->current_iteration / all_parameters_new.ode_solver_param.output_solution_every_x_steps;
    dg->output_results_vtk(file_number);

    //loop over time
    while(ode_solver->current_time < finalTime){
        //get timestep
        const double time_step =  get_timestep(dg,poly_degree, delta_x);
        if(ode_solver->current_iteration%all_parameters_new.ode_solver_param.print_iteration_modulo==0)
            pcout<<"time step "<<time_step<<" current time "<<ode_solver->current_time<<std::endl;
        //take the minimum timestep from all processors.
        const double dt = dealii::Utilities::MPI::min(time_step, mpi_communicator);
        //integrate in time
        ode_solver->step_in_time(dt, false);
        ode_solver->current_iteration += 1;
        //check if print solution
        const bool is_output_iteration = (ode_solver->current_iteration % all_parameters_new.ode_solver_param.output_solution_every_x_steps == 0);
        if (is_output_iteration) {
            const int file_number = ode_solver->current_iteration / all_parameters_new.ode_solver_param.output_solution_every_x_steps;
            dg->output_results_vtk(file_number);
        }
    }

    return 0;
}

#if PHILIP_DIM==2
    template class InviscidVortexShockInteraction <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

