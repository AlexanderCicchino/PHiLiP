#include <fstream>
#include "dg/dg_factory.hpp"
#include "euler_sine_wave.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "physics/initial_conditions/initial_condition_function.h"
#include "mesh/grids/straight_periodic_cube.hpp"
#include "mesh/grids/nonsymmetric_curved_periodic_grid.hpp"
#include "mesh/grids/nonsymmetric_curved_periodic_grid_chan.hpp"
#include "physics/exact_solutions/exact_solution.h"

#include <eigen/Eigen/Eigenvalues>
#include <eigen/Eigen/Dense>

#include <deal.II/base/convergence_table.h>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
EulerSineWave<dim, nstate>::EulerSineWave(const Parameters::AllParameters *const parameters_input)
    : TestsBase::TestsBase(parameters_input)
{}


template<int dim, int nstate>
double EulerSineWave<dim, nstate>::compute_kinetic_energy(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
{
    //returns the energy in the L2-norm (physically relevant)
    int overintegrate = 10 ;
    dealii::QGauss<dim> quad_extra(dg->max_degree+1+overintegrate);
    const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::FEValues<dim,dim> fe_values_extra(mapping, dg->fe_collection[poly_degree], quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::array<double,nstate> soln_at_q;

    double total_kinetic_energy = 0;

    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);

    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;

        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        //Please see Eq. 3.21 in Gassner, Gregor J., Andrew R. Winters, and David A. Kopriva. "Split form nodal discontinuous Galerkin schemes with summation-by-parts property for the compressible Euler equations." Journal of Computational Physics 327 (2016): 39-66.
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
            for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
             const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
            }

            const double density = soln_at_q[0];

            double quadrature_kinetic_energy = 0.0;
            for(int i=0;i<dim;i++){
                quadrature_kinetic_energy += 0.5*(soln_at_q[i+1]*soln_at_q[i+1])/density;
            }

            total_kinetic_energy += quadrature_kinetic_energy * fe_values_extra.JxW(iquad);
        }
    }
    return total_kinetic_energy;
}

template<int dim, int nstate>
double EulerSineWave<dim, nstate>::get_timestep(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree, const double delta_x) const
{
     //get local CFL
    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, dg->max_grid_degree);
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);
    const unsigned int n_shape_fns = n_dofs_cell / nstate;

    double cfl_min = 1e100;
    std::shared_ptr < Physics::PhysicsBase<dim, nstate, double > > pde_physics_double  = PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters);
    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;

        cell->get_dof_indices (dofs_indices);
        std::array<std::vector<double>,nstate> soln_coeff;
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0)
                soln_coeff[istate].resize(n_shape_fns);
            soln_coeff[istate][ishape] = dg->solution(dofs_indices[idof]);
        }

        std::array<std::vector<double>,nstate> soln_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }

        std::vector< double > convective_eigenvalues(n_quad_pts);
        for (unsigned int isol = 0; isol < n_quad_pts; ++isol) {
            std::array<double,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][isol];
            }
            convective_eigenvalues[isol] = pde_physics_double->max_convective_eigenvalue (soln_state);
        }
        const double max_eig = *(std::max_element(convective_eigenvalues.begin(), convective_eigenvalues.end()));

       // double cfl = 0.1 * delta_x/max_eig;
        double cfl = dg->all_parameters->flow_solver_param.courant_friedrichs_lewy_number * delta_x/max_eig;
        if(cfl < cfl_min)
            cfl_min = cfl;

    }
    return cfl_min;
}

template <int dim, int nstate>
void EulerSineWave<dim, nstate>::solve(std::shared_ptr<dealii::parallel::distributed::Triangulation<dim>> &grid,
                                               const unsigned int poly_degree, const unsigned int grid_degree,
                                               const double left, const double right,
                                               const unsigned int igrid, const unsigned int igrid_start, const unsigned int n_grids,
                                               std::array<double,2> &grid_size, std::array<double,2> &soln_error) const
{
    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;  

//    int CFL_flag = 1;
//    double CFL = 0.2;
//    CFL = 0.1;
 //   while(CFL_flag != 0){
       // CFL += 0.01;
//        all_parameters_new.flow_solver_param.courant_friedrichs_lewy_number = CFL;
//        pcout<<"CFL "<<CFL<<std::endl;

        // Create DG
        pcout<<"about to create DG"<<std::endl;
        std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
        pcout<<"created DG"<<std::endl;
        dg->allocate_system (false, false, false);
         
        pcout << "Implement initial conditions" << std::endl;
        std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function = 
                    InitialConditionFactory<dim,nstate,double>::create_InitialConditionFunction(&all_parameters_new);
        SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);

        dg->solution.update_ghost_values();
        const unsigned int n_global_active_cells2 = grid->n_global_active_cells();
        double delta_x = (right-left)/pow(n_global_active_cells2,1.0/dim)/(poly_degree+1.0);
        //const unsigned int n_cells_per_dim = pow(2.0,igrid)*n_cells_start;
        //double delta_x = (right-left)/n_cells_per_dim/(poly_degree+1.0);
        pcout<<" delta x "<<delta_x<<std::endl;
         
        all_parameters_new.ode_solver_param.initial_time_step =  get_timestep(dg,poly_degree,delta_x);
         
         pcout<<"got timestep"<<std::endl;
        std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
        const double finalTime = all_parameters_new.flow_solver_param.final_time;
       // const double finalTime = 2.0*all_parameters_new.ode_solver_param.initial_time_step;
        std::cout<<"Final time "<<finalTime<<std::endl;

        std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_double  = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters));
        ode_solver->current_iteration = 0;
        ode_solver->allocate_ode_system();
        pcout<<"allocated ode system"<<std::endl;


        while(ode_solver->current_time < finalTime){
            const double time_step =  get_timestep(dg,poly_degree, delta_x);
         //       const double M_infty_temp = sqrt(2.0/1.4);
         //       double time_step = 0.1 * delta_x / M_infty_temp;
            double dt = dealii::Utilities::MPI::min(time_step, mpi_communicator);
            if(ode_solver->current_time + dt > finalTime){
                dt = finalTime - ode_solver->current_time;
            }
            if(ode_solver->current_iteration%all_parameters_new.ode_solver_param.print_iteration_modulo==0)
                pcout<<"time step "<<time_step<<" current time "<<ode_solver->current_time<<std::endl;
           // pcout<<"doing step "<<time_step<<" current time "<<ode_solver->current_time<<std::endl;
            ode_solver->step_in_time(dt, false);
            ode_solver->current_iteration += 1;
            const bool is_output_iteration = (ode_solver->current_iteration % all_parameters_new.ode_solver_param.output_solution_every_x_steps == 0);
            if (is_output_iteration) {
                const int file_number = ode_solver->current_iteration / all_parameters_new.ode_solver_param.output_solution_every_x_steps;
                dg->output_results_vtk(file_number);
            }
        }
        pcout<<" current time end "<<ode_solver->current_time<<std::endl;

           // ode_solver->advance_solution_time(finalTime);
            const unsigned int n_global_active_cells = grid->n_global_active_cells();
            const unsigned int n_dofs = dg->dof_handler.n_dofs();
            pcout << "Dimension: " << dim
                  << "\t Polynomial degree p: " << poly_degree
                  << std::endl
                  << "Grid number: " << igrid-igrid_start+1 << "/" << n_grids-igrid_start
                  << ". Number of active cells: " << n_global_active_cells
                  << ". Number of degrees of freedom: " << n_dofs
                  << std::endl;

            // Overintegrate the error to make sure there is not integration error in the error estimate
            const int overintegrate = 10;
            dealii::QGauss<dim> quad_extra(poly_degree+1+overintegrate);
            dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
            std::array<double,nstate> soln_at_q;

            double l2error = 0.0;
            double l2error_density = 0.0;
            //generate exact solution at final time
            std::shared_ptr<ExactSolutionFunction<dim,nstate,double>> exact_solution_function;
           // exact_solution_function = ExactSolutionFactory<dim,nstate,double>::create_ExactSolutionFunction(all_parameters_new.flow_solver_param, finalTime);
            exact_solution_function = ExactSolutionFactory<dim,nstate,double>::create_ExactSolutionFunction(all_parameters_new.flow_solver_param, ode_solver->current_time);

            // Integrate solution error and output error
            std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
            for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {

                if (!cell->is_locally_owned()) continue;

                fe_values_extra.reinit (cell);
                cell->get_dof_indices (dofs_indices);

                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                    std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
                    for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                        const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                        soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                    }
                    const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));
                    //Compute Lp error for istate=0, which is density
                    std::array<double,nstate> exact_soln_at_q;
                    for (unsigned int istate = 0; istate < nstate; ++istate) { 
                        exact_soln_at_q[istate] = exact_solution_function->value(qpoint, istate);
                    }
                    const double pressure_exact = euler_double->compute_pressure(exact_soln_at_q);
                     
                    const double pressure = euler_double->compute_pressure(soln_at_q);
                    const double density = exact_soln_at_q[0];
                    l2error += pow(pressure - pressure_exact, 2) * fe_values_extra.JxW(iquad);
                    // l2error += abs(pressure - pressure_exact) * fe_values_extra.JxW(iquad);
                    l2error_density += pow(density - soln_at_q[0], 2) * fe_values_extra.JxW(iquad);
                }
            }
            const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, mpi_communicator));
          //  const double l2error_mpi_sum = dealii::Utilities::MPI::sum(l2error, mpi_communicator);
            const double l2error_mpi_sum_density = std::sqrt(dealii::Utilities::MPI::sum(l2error_density, mpi_communicator));

            // Convergence table
            const double dx = 1.0/pow(n_dofs/nstate,(1.0/dim));
            grid_size[igrid-igrid_start] = dx;
            soln_error[igrid-igrid_start] = l2error_mpi_sum_density;

            std::cout << std::setprecision(16) << std::fixed;
            pcout << " Grid size h: " << dx 
                  << " L2-pressure_error: " << l2error_mpi_sum
                  << " L2-density_error: " << l2error_mpi_sum_density
                  << " Residual: " << ode_solver->residual_norm
                  << std::endl;

            if (igrid > igrid_start) {
                const double slope_soln_err = log(soln_error[igrid-igrid_start]/soln_error[igrid-igrid_start-1])
                                      / log(grid_size[igrid-igrid_start]/grid_size[igrid-igrid_start-1]);
                pcout << "From grid " << igrid-igrid_start
                      << "  to grid " << igrid-igrid_start+1
                      << "  dimension: " << dim
                      << "  polynomial degree p: " << poly_degree
                      << std::endl
                      << "  solution_error1 " << soln_error[igrid-igrid_start-1]
                      << "  solution_error2 " << soln_error[igrid-igrid_start]
                      << "  slope " << slope_soln_err
                      << std::endl;
                if(igrid == n_grids-1){
                    if(std::abs(slope_soln_err-(poly_degree+1))>0.05){
                        pcout<<"Not get correct orders"<<std::endl;
                        std::abort();
                        //return 1;
                    }
                }
            }


#if 0
        pcout<<"difference pressure "<<abs(l2error_mpi_sum-0.0011168725921635)<<std::endl;
        if(abs(l2error_mpi_sum-0.0011168725921635)>=1e-3 ||abs(l2error_mpi_sum_density-0.0011162711985459)>=1e-3){
            pcout<<"difference pressure "<<abs(l2error_mpi_sum-0.0011168725921635)<<std::endl;
            pcout<<"difference density "<<abs(l2error_mpi_sum_density-0.0011162711985459)<<std::endl;
            CFL_flag = 0;
            pcout<<"CFL max "<<CFL-0.01<<std::endl;
        }
#endif
#if 0
        //c+ 2D GL
        pcout<<"difference pressure "<<abs(l2error_mpi_sum-0.0069677942066219)<<std::endl;
        if(abs(l2error_mpi_sum-0.0069677942066219)>=1e-3 ||abs(l2error_mpi_sum_density-0.0076676962109569)>=1e-3){
            pcout<<"difference pressure "<<abs(l2error_mpi_sum-0.0069677942066219)<<std::endl;
            pcout<<"difference density "<<abs(l2error_mpi_sum_density-0.0076676962109569)<<std::endl;
            CFL_flag = 0;
            pcout<<"CFL max "<<CFL-0.01<<std::endl;
        }
#endif

//        pcout<<"difference pressure "<<abs(l2error_mpi_sum-0.7448712945470424)<<std::endl;
//        if(abs(l2error_mpi_sum-0.7448712945470424)>=1e-3 ||abs(l2error_mpi_sum_density-0.5927959282730064)>=1e-3){
//            pcout<<"difference pressure "<<abs(l2error_mpi_sum-0.7448712945470424)<<std::endl;
//            pcout<<"difference density "<<abs(l2error_mpi_sum_density-0.5927959282730064)<<std::endl;
//            CFL_flag = 0;
//            pcout<<"CFL max "<<CFL-0.01<<std::endl;
//        }
   // }
        
}

template <int dim, int nstate>
int EulerSineWave<dim, nstate>::run_test() const
{

    using real = double;

    const double left  = all_parameters->flow_solver_param.grid_left_bound;
    const double right = all_parameters->flow_solver_param.grid_right_bound;
    const unsigned int poly_degree= all_parameters->flow_solver_param.poly_degree;
//    const unsigned int igrid_start = all_parameters->flow_solver_param.number_of_grid_elements_per_dimension;
    std::array<double,2> grid_size;
    std::array<double,2> soln_error;
    const unsigned int n_cells_start = all_parameters->flow_solver_param.number_of_grid_elements_per_dimension;
    const unsigned int igrid_start = 0;
    const unsigned int n_grids = 4;

pcout<<"left "<<left<<" right "<<right<<std::endl;
pcout<<"igrd start is "<<igrid_start<<std::endl;

    for(unsigned int igrid = igrid_start; igrid<n_grids; igrid++){
        using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
            MPI_COMM_WORLD,
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));
        pcout<<"got triangulation "<<std::endl;

        const unsigned int n_cells_per_dim = pow(2.0,igrid)*n_cells_start;
       // const unsigned int n_cells_per_dim = 2.0*pow(2.0,igrid) + n_cells_start;

        const unsigned int grid_degree = all_parameters->use_curvilinear_grid ? poly_degree : 1;
        if(all_parameters->use_curvilinear_grid){
            //if curvilinear
           // PHiLiP::Grids::nonsymmetric_curved_grid<dim,Triangulation>(*grid, n_refinements);
       //     PHiLiP::Grids::nonsymmetric_curved_grid<dim,Triangulation>(*grid, igrid, true, left, right);

        //    if(dim==3){
                PHiLiP::Grids::nonsymmetric_curved_grid<dim,Triangulation>(*grid, log(n_cells_per_dim)/log(2.0), true, left, right);
         //   }
         //   else{
         //       PHiLiP::Grids::nonsymmetric_curved_grid_chan<dim,Triangulation>(*grid, n_cells_per_dim);
         //   }
        }
        else{
            //if straight
           // PHiLiP::Grids::straight_periodic_cube<dim,Triangulation>(grid, left, right, pow(2.0, igrid));
            const double x_right = right;
            const double y_right = right;
            const double z_right = right;
            const bool colorize = true;
            std::vector<unsigned int> repititions(dim);
            dealii::Point<dim> point1;
            dealii::Point<dim> point2;
            for(int idim=0; idim<dim; idim++){
                repititions[idim] = n_cells_per_dim;
                point1[idim] = left;
                if(idim==0)
                    point2[idim] = x_right;
                if(idim==1)
                    point2[idim] = y_right;
                if(idim==2)
                    point2[idim] = z_right;
            }
            pcout<<"making grid"<<std::endl;
            dealii::GridGenerator::subdivided_hyper_rectangle (*grid, repititions, point1, point2, colorize);
            pcout<<"made grid"<<std::endl;
            std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::parallel::distributed::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
            dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
            if(dim>=2) dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
            if(dim>=3) dealii::GridTools::collect_periodic_faces(*grid,4,5,2,matched_pairs);
            grid->add_periodicity(matched_pairs);
        }
        pcout<<"did warping"<<std::endl;
       // dealii::GridGenerator::hyper_cube(*grid, left, right, true);
//        pcout<<"generated grid "<<std::endl;
//        std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::parallel::distributed::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
//        dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
//        if(dim>=2) dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
//        if(dim>=3) dealii::GridTools::collect_periodic_faces(*grid,4,5,2,matched_pairs);
//        grid->add_periodicity(matched_pairs);
//        grid->refine_global(igrid);
        this->solve(grid, poly_degree, grid_degree, left, right, igrid, igrid_start, n_grids, grid_size, soln_error);
    }//end of grid loop


    return 0;
}

#if PHILIP_DIM>1
    template class EulerSineWave <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace


