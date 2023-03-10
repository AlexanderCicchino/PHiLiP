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

#include "burgers_linear_stability.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "dg/dg.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_base.h"
#include <fstream>
#include "ode_solver/ode_solver_factory.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "physics/initial_conditions/initial_condition_function.h"

#include <eigen/Eigen/Eigenvalues>
#include <eigen/Eigen/Dense>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
BurgersLinearStability<dim, nstate>::BurgersLinearStability(const PHiLiP::Parameters::AllParameters *const parameters_input)
: TestsBase::TestsBase(parameters_input)
{}

template <int dim, int nstate>
int BurgersLinearStability<dim, nstate>::run_test() const
{
    pcout << " Running Burgers linear stability. " << std::endl;

    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;  
   double left  = -1.0;
   double right =  1.0;
  //  double left  = 0.0;
  //  double right = 2.0;
    const unsigned int n_grids = 4;
   // const unsigned int n_grids = 5;
   // const unsigned int n_grids = 2;
    std::vector<double> grid_size(n_grids);
    std::vector<double> soln_error(n_grids);
    unsigned int poly_degree = 4;
    //unsigned int poly_degree = 8;
  //  unsigned int poly_degree = 15;
    dealii::ConvergenceTable convergence_table;
    const unsigned int igrid_start = 3;
   // const unsigned int igrid_start = 4;
  //  const unsigned int igrid_start = 1;
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
        dg->allocate_system ();
         
        //initialize IC
        pcout<<"Setting up Initial Condition"<<std::endl;
        // Create initial condition function
        std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function = 
            InitialConditionFactory<dim,nstate,double>::create_InitialConditionFunction(&all_parameters_new);
        SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);

        Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver;

        // Create ODE solver using the factory and providing the DG object
        std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
        ode_solver->allocate_ode_system();
        double dt = all_parameters_new.ode_solver_param.initial_time_step;
       // double finalTime = 100.0*dt;
       // double finalTime = 0.3;
      //  double finalTime = 0.9;
       // double finalTime = dt;
        double finalTime=0.0;
        for (int itime = 0; itime <= std::ceil(finalTime/dt); ++ itime){
          //  ode_solver->step_in_time(dt, false);
//        }
    std::cout<<" time "<<ode_solver->current_time<<std::endl;
  //  }

        //Perform perturbation
//        std::vector<std::vector<double>> dRdU(dg->solution.size());
        Eigen::MatrixXd dRdU(dg->solution.size(), dg->solution.size());
//        std::vector<double> col_dRdU(dg->solution.size());
        dealii::LinearAlgebra::distributed::Vector<double> col_dRdU(dg->right_hand_side.size());
        const double perturbation = 1e-8;
        std::cout<<"doing perturbations"<<std::endl;
        for(unsigned int eig_direction=0; eig_direction<dg->solution.size(); eig_direction++){
            double solution_init_value = dg->solution[eig_direction];
            dg->solution[eig_direction] += perturbation;
            dg->assemble_residual();
         
//        SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);
         
            if(all_parameters_new.use_inverse_mass_on_the_fly){
                dg->apply_inverse_global_mass_matrix(dg->right_hand_side, col_dRdU); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
            } else{
                dg->global_inverse_mass_matrix.vmult(col_dRdU, dg->right_hand_side); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
            }
            for(unsigned int i=0; i<dg->solution.size(); i++){
                dRdU(i,eig_direction) = col_dRdU[i];
            }
            dg->solution[eig_direction] -= 2.0 * perturbation;
            dg->assemble_residual();
            if(all_parameters_new.use_inverse_mass_on_the_fly){
                dg->apply_inverse_global_mass_matrix(dg->right_hand_side, col_dRdU); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
            } else{
                dg->global_inverse_mass_matrix.vmult(col_dRdU, dg->right_hand_side); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
            }
            for(unsigned int i=0; i<dg->solution.size(); i++){
                dRdU(i,eig_direction) -= col_dRdU[i];
                dRdU(i,eig_direction) /= (2.0 * perturbation);
            }
            dg->solution[eig_direction] = solution_init_value;//set back to the IC
         
//        SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);
        }
    std::cout<<"got perts"<<std::endl;
//    Eigen::MatrixXd dRdU_mpi(dg->solution.size(), dg->solution.size());
//    for(unsigned int i=0; i<dg->solution.size(); i++){
//        for(unsigned int j=0; j<dg->solution.size(); j++){
//            dRdU_mpi(i,j) = dealii::Utilities::MPI::sum(dRdU(i,j), this->mpi_communicator);
//        }
//    }
    eigen_solver.compute(dRdU);

        for(unsigned int i=0; i<eigen_solver.eigenvalues().size(); i++){
            if(eigen_solver.eigenvalues()[i].real() > 1e-6){
                pcout<<"time "<<ode_solver->current_time<<" eigenvalue "<<eigen_solver.eigenvalues()[i].real()<<std::endl;
//                Eigen::VectorXcd eigen_vect = eigen_solver.eigenvectors().col(i);
//                pcout<< "the eigenvector "<<std::endl<<std::fixed << std::setprecision(16) << eigen_vect<< std::fixed << std::endl;
//        pcout<<"the DRDU"<<std::endl;
//            for(unsigned int i=0; i<dg->solution.size(); i++){
//            for(unsigned int j=0; j<dg->solution.size(); j++){
//                pcout<<dRdU(i,j)<<" ";
//            }
//            pcout<<std::endl;
//            }
            }
        }

//        pcout<<"the DRDU"<<std::endl;
//            for(unsigned int i=0; i<dg->solution.size(); i++){
//            for(unsigned int j=0; j<dg->solution.size(); j++){
//                pcout<<dRdU(i,j)<<" ";
//            }
//            pcout<<std::endl;
//            }

        }//end ode solver



        //Print to a file the eigenvalues vs x to plot
       // std::ofstream myfile ("dRdU_burgers.gpl" , std::ios::trunc);
        std::ofstream myfile ("computed_eigenvalues_burgers_play_around_new.gpl" , std::ios::trunc);
//        for(unsigned int i=0; i<dg->solution.size(); i++){
//            for(unsigned int j=0; j<dg->solution.size(); j++){
                std::cout << std::setprecision(16) << std::fixed;
               // myfile<< std::fixed << std::setprecision(16) << dRdU[j][i] << " "<< std::fixed;
                myfile<< std::fixed << std::setprecision(16) << eigen_solver.eigenvalues() << " "<< std::fixed<<std::endl;
//            }
//            myfile<< std::endl;
//        }

        myfile.close();

        std::ofstream myfile3 ("computed_eigenvectors_burgers_play_around.gpl" , std::ios::trunc);
        std::cout << std::setprecision(16) << std::fixed;
        myfile3<< std::fixed << std::setprecision(16) << eigen_solver.eigenvectors()<< std::fixed << std::endl<<std::endl;
        myfile3.close();

        std::ofstream myfile4 ("positive_eigen_pairs.gpl" , std::ios::trunc);
        std::cout << std::setprecision(16) << std::fixed;
        for(unsigned int i=0; i<eigen_solver.eigenvalues().size(); i++){
            if(eigen_solver.eigenvalues()[i].real() > 1e-7){
                myfile4<< "the eigenvalue "<<std::endl<<std::fixed << std::setprecision(16) << eigen_solver.eigenvalues()[i]<< std::fixed << std::endl;
                Eigen::VectorXcd eigen_vect = eigen_solver.eigenvectors().col(i);
                myfile4<< "the eigenvector "<<std::endl<<std::fixed << std::setprecision(16) << eigen_vect<< std::fixed << std::endl;
                Eigen::VectorXcd eigen_vect_row = eigen_solver.eigenvectors().inverse().row(i);
                myfile4<< "the eigenvector row "<<std::endl<<std::fixed << std::setprecision(16) << eigen_vect_row<< std::fixed << std::endl;
            }
        }
        myfile4.close();


        dealii::QGaussLobatto<dim> quad_extra(dg->max_degree+1);
        dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], quad_extra, 
                dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
        const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
        std::array<double,nstate> soln_at_q;
        std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
        
        //Print to a file the final solution vs x to plot
        std::ofstream myfile2 ("solution_burgers.gpl" , std::ios::trunc);

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
    }//end of grid loop
    return 0; //if got to here means passed the test, otherwise would've failed earlier
}

#if PHILIP_DIM==1
template class BurgersLinearStability<PHILIP_DIM,PHILIP_DIM>;
#endif

} // Tests namespace
} // PHiLiP namespace

