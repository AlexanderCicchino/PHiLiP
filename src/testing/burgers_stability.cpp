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
#include <deal.II/base/convergence_table.h>

#include "burgers_stability.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "dg/dg.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_base.h"
#include <fstream>
#include "ode_solver/ode_solver_factory.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "physics/initial_conditions/initial_condition_function.h"
#include "physics/burgers.h"


namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
BurgersEnergyStability<dim, nstate>::BurgersEnergyStability(const PHiLiP::Parameters::AllParameters *const parameters_input)
: TestsBase::TestsBase(parameters_input)
{}

template<int dim, int nstate>
std::array<double,2> BurgersEnergyStability<dim, nstate>::compute_change_in_entropy(const std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
{
    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    //We have to project the vector of entropy variables because the mass matrix has an interpolation from solution nodes built into it.
    OPERATOR::vol_projection_operator<dim,2*dim> vol_projection(1, poly_degree, dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, dg->max_grid_degree);
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global(dg->right_hand_side);
    dealii::LinearAlgebra::distributed::Vector<double> energy_var_hat_global(dg->right_hand_side);
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

    std::shared_ptr < Physics::Burgers<dim, nstate, double > > burgers_double  = std::dynamic_pointer_cast<Physics::Burgers<dim,dim,double>>(PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters));

    dg->assemble_residual();
    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);

        //get solution modal coeff
        std::array<std::vector<double>,nstate> soln_coeff;
        std::array<std::vector<double>,nstate> rhs_coeff;
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0){
                soln_coeff[istate].resize(n_shape_fns);
                rhs_coeff[istate].resize(n_shape_fns);
            }
            soln_coeff[istate][ishape] = dg->solution(dofs_indices[idof]);
            rhs_coeff[istate][ishape] = dg->right_hand_side(dofs_indices[idof]);
        }

        //interpolate solution to quadrature points
        std::array<std::vector<double>,nstate> soln_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }
        //compute entropy and kinetic energy "entropy" variables at quad points
        std::array<std::vector<double>,nstate> entropy_var_at_q;
        std::array<std::vector<double>,nstate> energy_var_at_q;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<double,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }
            std::array<double,nstate> entropy_var_state = burgers_double->compute_entropy_variables(soln_state);
            std::array<double,nstate> kin_energy_state = burgers_double->compute_entropy_variables(soln_state);
            for(int istate=0; istate<nstate; istate++){
                if(iquad==0){
                    entropy_var_at_q[istate].resize(n_quad_pts);
                    energy_var_at_q[istate].resize(n_quad_pts);
                }
                energy_var_at_q[istate][iquad] = kin_energy_state[istate];
                entropy_var_at_q[istate][iquad] = entropy_var_state[istate];
            }
        }
        //project the enrtopy and KE var to modal coefficients
        //then write it into a global vector
   //     double cell_entropy = 0.0;
        for(int istate=0; istate<nstate; istate++){
            //Projected vector of entropy variables.
            std::vector<double> entropy_var_hat(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(entropy_var_at_q[istate], entropy_var_hat,
                                                 vol_projection.oneD_vol_operator);
            std::vector<double> energy_var_hat(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(energy_var_at_q[istate], energy_var_hat,
                                                 vol_projection.oneD_vol_operator);

            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                const unsigned int idof = istate * n_shape_fns + ishape;
                entropy_var_hat_global[dofs_indices[idof]] = entropy_var_hat[ishape];
                energy_var_hat_global[dofs_indices[idof]] = energy_var_hat[ishape];

    //            cell_entropy += entropy_var_hat[ishape] * rhs_coeff[istate][ishape];
            }
        }
//        if(cell_entropy>0){
    //        std::cout<<"cell entropy pos "<<cell_entropy<<std::endl;
//        }
    }

    //evaluate the change in entropy and change in KE
    std::array<double,2> change_entropy_and_energy;
    change_entropy_and_energy[0] = entropy_var_hat_global * dg->right_hand_side;
    change_entropy_and_energy[1] = energy_var_hat_global * dg->right_hand_side;

//    if(dg->all_parameters->use_vanishing_viscosity){
//        for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
//            if (!cell->is_locally_owned()) continue;
//            cell->get_dof_indices (dofs_indices);
//            
//            const double visc_coeff = dg->cell_entropy_production_coeff[dofs_indices[0]];
//            for(int idim=0; idim<dim; idim++){
//                for(unsigned int idof=0; idof<n_dofs_cell; idof++){
//                    change_entropy_and_energy[0] += dg->auxiliary_solution[idim][dofs_indices[idof]]
//                                                  * dg->auxiliary_right_hand_side[idim][dofs_indices[idof]]
//                                                  * visc_coeff;
//                }
//            }
//         
//        }
//    }

    return change_entropy_and_energy;
}
template<int dim, int nstate>
double BurgersEnergyStability<dim, nstate>::compute_energy(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg) const
{
    double energy = 0.0;
    dealii::LinearAlgebra::distributed::Vector<double> mass_matrix_times_solution(dg->right_hand_side);
    if(dg->all_parameters->use_inverse_mass_on_the_fly)
        dg->apply_global_mass_matrix(dg->solution,mass_matrix_times_solution);
    else
        dg->global_mass_matrix.vmult( mass_matrix_times_solution, dg->solution);
    //Since we normalize the energy later, don't bother scaling by 0.5
    //Energy \f$ = 0.5 * \int u^2 d\Omega_m \f$
    energy = dg->solution * mass_matrix_times_solution;
    
    return energy;
}

template<int dim, int nstate>
double BurgersEnergyStability<dim, nstate>::compute_conservation(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg, const double poly_degree) const
{
    // Conservation \f$ =  \int 1 * u d\Omega_m \f$
    double conservation = 0.0;
    dealii::LinearAlgebra::distributed::Vector<double> mass_matrix_times_solution(dg->right_hand_side);
    if(dg->all_parameters->use_inverse_mass_on_the_fly)
        dg->apply_global_mass_matrix(dg->solution,mass_matrix_times_solution);
    else
        dg->global_mass_matrix.vmult( mass_matrix_times_solution, dg->solution);

    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    std::vector<double> ones(n_quad_pts, 1.0);
    //Projected vector of ones. That is, the interpolation of ones_hat to the volume nodes is 1.
    std::vector<double> ones_hat(n_dofs_cell);
    //We have to project the vector of ones because the mass matrix has an interpolation from solution nodes built into it.
    OPERATOR::vol_projection_operator<dim,2*dim> vol_projection(dg->nstate, poly_degree, dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(dg->oneD_fe_collection[poly_degree], dg->oneD_quadrature_collection[poly_degree]);
    vol_projection.matrix_vector_mult_1D(ones, ones_hat,
                                               vol_projection.oneD_vol_operator);

    dealii::LinearAlgebra::distributed::Vector<double> ones_hat_global(dg->right_hand_side);
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);
    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);
        for(unsigned int idof=0;idof<n_dofs_cell; idof++){
            ones_hat_global[dofs_indices[idof]] = ones_hat[idof];
        }
    }

    conservation = ones_hat_global * mass_matrix_times_solution;

    return conservation;
}
template<int dim, int nstate>
double BurgersEnergyStability<dim, nstate>::get_timestep(const std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree, const double delta_x) const
{
    dg->apply_entropy_production_correction();
    double max_diff_coeff = 0.0;
    std::vector<dealii::types::global_dof_index> dofs_indices (poly_degree + 1);
    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;

        cell->get_dof_indices (dofs_indices);
        double local_diff_coeff = dg->cell_entropy_production_coeff[dofs_indices[0]];
        if(local_diff_coeff > max_diff_coeff)
            max_diff_coeff = local_diff_coeff;
    }
    pcout<<"max diff coeff "<<max_diff_coeff<<std::endl;
    double timestep = dg->all_parameters->flow_solver_param.courant_friedrichs_lewy_number 
                    * delta_x*delta_x 
                    / max_diff_coeff;

    return timestep;
}

template <int dim, int nstate>
int BurgersEnergyStability<dim, nstate>::run_test() const
{
    pcout << " Running Burgers energy stability. " << std::endl;

    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;  
    double left = 0.0;
    double right = 2.0;
   // double left = -1.0;
   // double right = 1.0;
    const unsigned int n_grids = (all_parameters_new.use_energy) ? 4 : 6;
    std::vector<double> grid_size(n_grids);
    std::vector<double> soln_error(n_grids);
    unsigned int poly_degree = 4;
    dealii::ConvergenceTable convergence_table;
    const unsigned int igrid_start = 3;
    const unsigned int grid_degree = 1;

    for(unsigned int igrid = igrid_start; igrid<n_grids; igrid++){

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
        //straight grid setup
        dealii::GridGenerator::hyper_cube(*grid, left, right, true);
        //found the periodicity in dealii doesn't work as expected in 1D so I hard coded the 1D periodic condition in DG
#if PHILIP_DIM==1
        std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
        dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
        grid->add_periodicity(matched_pairs);
#else
        std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::parallel::distributed::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
        dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
        if(dim>=2) dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
        if(dim>=3) dealii::GridTools::collect_periodic_faces(*grid,4,5,2,matched_pairs);
        grid->add_periodicity(matched_pairs);
#endif
        grid->refine_global(igrid);
        pcout << "Grid generated and refined" << std::endl;
        //CFL number
        const unsigned int n_global_active_cells2 = grid->n_global_active_cells();
        double n_dofs_cfl = pow(n_global_active_cells2,dim) * pow(poly_degree+1.0, dim);
        double delta_x = (right-left)/pow(n_dofs_cfl,(1.0/dim)); 
        all_parameters_new.ode_solver_param.initial_time_step =  0.5*delta_x;
        //use 0.0001 to be consisitent with Ranocha and Gassner papers
        all_parameters_new.ode_solver_param.initial_time_step =  0.0001;
        
        //allocate dg
        std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
        pcout << "dg created" <<std::endl;
        dg->allocate_system (false,false,false);
         
        //initialize IC
        pcout<<"Setting up Initial Condition"<<std::endl;
        // Create initial condition function
        std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function = 
            InitialConditionFactory<dim,nstate,double>::create_InitialConditionFunction(&all_parameters_new);
        SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);

        // Create ODE solver using the factory and providing the DG object
        std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);

        double finalTime = 3.0;

        if (all_parameters_new.use_energy == true){//for split form get energy

       // double dt = all_parameters_new.ode_solver_param.initial_time_step;

        // need to call ode_solver before calculating energy because mass matrix isn't allocated yet.
        ode_solver->current_iteration = 0;
        ode_solver->advance_solution_time(0.000001);
        double initial_energy = compute_energy(dg);
        double initial_conservation = compute_conservation(dg, poly_degree);

        // currently the only way to calculate energy at each time-step is to advance solution by dt instead of finaltime
        // this causes some issues with outputs (only one file is output, which is overwritten at each time step)
        // also the ode solver output doesn't make sense (says "iteration 1 out of 1")
        // but it works. I'll keep it for now and need to modify the output functions later to account for this.
        std::ofstream myfile ("energy_plot_burgers.gpl" , std::ios::trunc);
         
        ode_solver->current_iteration = 0;
      //  for (int i = 0; i < std::ceil(finalTime/dt); ++ i)
       // {
        while(ode_solver->current_time < finalTime){
        //        const double time_step =  get_timestep(dg,poly_degree, delta_x);
         //       pcout<<"timstep "<<time_step<<std::endl;
                const double time_step = 0.0001;
                ode_solver->advance_solution_time(time_step);
                //Energy
                double current_energy = compute_energy(dg);
                current_energy /=initial_energy;
                std::cout << std::setprecision(16) << std::fixed;
                pcout << "Energy at time " << ode_solver->current_time << " is " << current_energy << std::endl;
                myfile << ode_solver->current_time << " " << std::fixed << std::setprecision(16) << current_energy << std::endl;
                if (current_energy*initial_energy - initial_energy >= 1.0)
                {
                    pcout<<"Energy not monotonicaly decreasing"<<std::endl;
                    return 1;
                    break;
                }
//                if ( (current_energy*initial_energy - initial_energy >= 1.0e-11)&&(all_parameters_new.conv_num_flux_type == Parameters::AllParameters::ConvectiveNumericalFlux::two_point_flux) )
//                {
//                    pcout<<"Energy not conserved"<<std::endl;
//                    return 1;
//                    break;
//                }
                //Conservation
                double current_conservation = compute_conservation(dg, poly_degree);
                current_conservation /=initial_conservation;
                std::cout << std::setprecision(16) << std::fixed;
                pcout << "Normalized Conservation at time " << ode_solver->current_time << " is " << current_conservation<< std::endl;
                myfile << ode_solver->current_time << " " << std::fixed << std::setprecision(16) << current_conservation << std::endl;
                if (current_conservation*initial_conservation - initial_conservation >= 10.00)
                {
                    pcout << "Not conserved" << std::endl;
                    return 1;
                    break;
                }
                const std::array<double,2> current_change_entropy = compute_change_in_entropy(dg, poly_degree);
                const double current_change_entropy_mpi = dealii::Utilities::MPI::sum(current_change_entropy[0], mpi_communicator);
                const double current_change_energy_mpi = dealii::Utilities::MPI::sum(current_change_entropy[1], mpi_communicator);
                //write to the file the change in entropy mpi
               // myfile<<ode_solver->current_time<<" "<< current_change_entropy_mpi <<std::endl;
                pcout << "M plus K norm Change in Entropy at time " << ode_solver->current_time << " is " << current_change_entropy_mpi<< std::endl;
                pcout << "M plus K norm Change in Kinetic Energy at time " << ode_solver->current_time << " is " << current_change_energy_mpi<< std::endl;
            }
            myfile.close();
             
            //Print to a file the final solution vs x to plot
            std::ofstream myfile2 ("solution_burgers.gpl" , std::ios::trunc);
             
            dealii::QGaussLobatto<dim> quad_extra(dg->max_degree+1);
            dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
            std::array<double,nstate> soln_at_q;
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

                    std::cout << std::setprecision(16) << std::fixed;
                    myfile2<< std::fixed << std::setprecision(16) << qpoint[0] << std::fixed << std::setprecision(16) <<" " << soln_at_q[0]<< std::endl;
                }
            }
            myfile2.close();
        }//end of energy
        else{//do OOA
            finalTime = 0.001;//This is sufficient for verification
         //   finalTime = 0.5;

            ode_solver->current_iteration = 0;
            ode_solver->allocate_ode_system();

            while(ode_solver->current_time < finalTime){
                const double time_step = 0.0001;
                ode_solver->step_in_time(time_step,false);
                ode_solver->current_iteration += 1;
                if(ode_solver->current_iteration%all_parameters_new.ode_solver_param.print_iteration_modulo==0)
                    pcout<<"time step "<<time_step<<" current time "<<ode_solver->current_time<<std::endl;
            }
           // ode_solver->advance_solution_time(finalTime);
            const unsigned int n_global_active_cells = grid->n_global_active_cells();
            const unsigned int n_dofs = dg->dof_handler.n_dofs();
            pcout << "Dimension: " << dim
                  << "\t Polynomial degree p: " << poly_degree
                  << std::endl
                  << "Grid number: " << igrid+1 << "/" << n_grids
                  << ". Number of active cells: " << n_global_active_cells
                  << ". Number of degrees of freedom: " << n_dofs
                  << std::endl;

            // Overintegrate the error to make sure there is not integration error in the error estimate
            int overintegrate = 10;
            dealii::QGauss<dim> quad_extra(poly_degree+1+overintegrate);
            dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
            std::array<double,nstate> soln_at_q;

            double l2error = 0.0;

            // Integrate solution error and output error
            const double pi = atan(1)*4.0;
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

                    for (int istate=0; istate<nstate; ++istate) {
                    const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));
                    double uexact = 0.0;
                    for(int idim=0; idim<dim; idim++){
                       // uexact += cos(pi*(qpoint[idim]-finalTime));//for grid 1-3
                        uexact += cos(pi*(qpoint[idim]-ode_solver->current_time));//for grid 1-3
                    }
                        l2error += pow(soln_at_q[istate] - uexact, 2) * fe_values_extra.JxW(iquad);
                    }
                }
            }
            const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, mpi_communicator));

            // Convergence table
            const double dx = 1.0/pow(n_dofs,(1.0/dim));
            grid_size[igrid] = dx;
            soln_error[igrid] = l2error_mpi_sum;

            convergence_table.add_value("p", poly_degree);
            convergence_table.add_value("cells", n_global_active_cells);
            convergence_table.add_value("DoFs", n_dofs);
            convergence_table.add_value("dx", dx);
            convergence_table.add_value("soln_L2_error", l2error_mpi_sum);

            pcout << " Grid size h: " << dx 
                  << " L2-soln_error: " << l2error_mpi_sum
                  << " Residual: " << ode_solver->residual_norm
                  << std::endl;

            if (igrid > igrid_start) {
                const double slope_soln_err = log(soln_error[igrid]/soln_error[igrid-1])
                                      / log(grid_size[igrid]/grid_size[igrid-1]);
                pcout << "From grid " << igrid
                      << "  to grid " << igrid+1
                      << "  dimension: " << dim
                      << "  polynomial degree p: " << poly_degree
                      << std::endl
                      << "  solution_error1 " << soln_error[igrid-1]
                      << "  solution_error2 " << soln_error[igrid]
                      << "  slope " << slope_soln_err
                      << std::endl;
                if(igrid == n_grids-1){
                    if(std::abs(slope_soln_err-(poly_degree+1))>0.05){
                        return 1;
                    }
                }
            }
        
            pcout << " ********************************************"
                  << std::endl
                  << " Convergence rates for p = " << poly_degree
                  << std::endl
                  << " ********************************************"
                  << std::endl;
            convergence_table.evaluate_convergence_rates("soln_L2_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
            convergence_table.set_scientific("dx", true);
            convergence_table.set_scientific("soln_L2_error", true);
            if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());
        }//end of OOA
    }//end of grid loop
    return 0; //if got to here means passed the test, otherwise would've failed earlier
}

#if PHILIP_DIM==1
template class BurgersEnergyStability<PHILIP_DIM,PHILIP_DIM>;
#endif

} // Tests namespace
} // PHiLiP namespace
