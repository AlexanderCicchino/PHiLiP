#include <fstream>
#include "dg/dg_factory.hpp"
#include "euler_isentropic_vortex.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "physics/initial_conditions/initial_condition_function.h"
#include "mesh/grids/straight_periodic_cube.hpp"

#include <eigen/Eigen/Eigenvalues>
#include <eigen/Eigen/Dense>

#include <deal.II/base/convergence_table.h>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
EulerIsentropicVortex<dim, nstate>::EulerIsentropicVortex(const Parameters::AllParameters *const parameters_input)
    : TestsBase::TestsBase(parameters_input)
{}


template<int dim, int nstate>
double EulerIsentropicVortex<dim, nstate>::compute_kinetic_energy(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
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
double EulerIsentropicVortex<dim, nstate>::get_timestep(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree, const double delta_x) const
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
       // double cfl = 0.05 * delta_x/max_eig;
        double cfl = 0.1 * delta_x/max_eig;

       // double cfl = 0.000005 * delta_x/max_eig;
        if(cfl < cfl_min)
            cfl_min = cfl;
    }
    return cfl_min;
}

template <int dim, int nstate>
void EulerIsentropicVortex<dim, nstate>::solve(std::shared_ptr<dealii::parallel::distributed::Triangulation<dim>> &grid,
                                               const unsigned int poly_degree, const unsigned int grid_degree,
                                               const double left, const double right,
                                               const unsigned int igrid, const unsigned int igrid_start, const unsigned int n_grids,
                                               std::array<double,2> &grid_size, std::array<double,2> &soln_error) const
{
    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;  

        // Create DG
        std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
        pcout<<"created DG"<<std::endl;
        dg->allocate_system ();
         
        pcout << "Implement initial conditions" << std::endl;
        std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function = 
                    InitialConditionFactory<dim,nstate,double>::create_InitialConditionFunction(&all_parameters_new);
        SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);

        dg->solution.update_ghost_values();

#if 0
    //Do eigenvalues

    std::cout<<"doing eig"<<std::endl;
    Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver;
    Eigen::MatrixXd dRdU(dg->solution.size(), dg->solution.size());
    std::cout<<" allocating col DRdU"<<std::endl;
    dealii::LinearAlgebra::distributed::Vector<double> col_dRdU(dg->right_hand_side.size());
//    dealii::LinearAlgebra::distributed::Vector<double> col_dRdU;
//    col_dRdU.reinit(dg->locally_owned_dofs, dg->ghost_dofs, MPI_COMM_WORLD);
    const double perturbation = 1e-8;
    std::cout<<"doing perturbations"<<std::endl;
    for(unsigned int eig_direction=0; eig_direction<dg->solution.size(); eig_direction++){
        double solution_init_value = dg->solution[eig_direction];
        dg->solution[eig_direction] += perturbation;
        dg->assemble_residual();

//    SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);

//        if(all_parameters_new.use_inverse_mass_on_the_fly){
//            dg->apply_inverse_global_mass_matrix(dg->right_hand_side, col_dRdU); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
//        } else{
//            dg->global_inverse_mass_matrix.vmult(col_dRdU, dg->right_hand_side); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
//        }
        for(unsigned int i=0; i<dg->solution.size(); i++){
           // dRdU[i][eig_direction] = dg->right_hand_side[i];
           col_dRdU[i] = dg->right_hand_side[i];
            dRdU(i,eig_direction) = col_dRdU[i];
        }
        dg->solution[eig_direction] -= 2.0 * perturbation;
        dg->assemble_residual();
//        if(all_parameters_new.use_inverse_mass_on_the_fly){
//            dg->apply_inverse_global_mass_matrix(dg->right_hand_side, col_dRdU); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
//        } else{
//            dg->global_inverse_mass_matrix.vmult(col_dRdU, dg->right_hand_side); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
//        }
        for(unsigned int i=0; i<dg->solution.size(); i++){
           // dRdU[i][eig_direction] -= dg->right_hand_side[i];
           col_dRdU[i] = dg->right_hand_side[i];
            dRdU(i,eig_direction) -= col_dRdU[i];
            dRdU(i,eig_direction) /= (2.0 * perturbation);
        }
        dg->solution[eig_direction] = solution_init_value;//set back to the IC

//    SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);
    }
    std::cout<<"got perts"<<std::endl;
//    Eigen::MatrixXd dRdU_mpi(dg->solution.size(), dg->solution.size());
//    for(unsigned int i=0; i<dg->solution.size(); i++){
//        for(unsigned int j=0; j<dg->solution.size(); j++){
//            dRdU_mpi(i,j) = dealii::Utilities::MPI::sum(dRdU(i,j), this->mpi_communicator);
//        }
//    }
    eigen_solver.compute(dRdU);
   // eigen_solver.compute(dRdU_mpi);
    std::cout<<"got eigs"<<std::endl;
    std::ofstream myfile2 ("computed_eigenvalues_euler_play_around.gpl" , std::ios::trunc);
    std::cout << std::setprecision(16) << std::fixed;
    myfile2<< std::fixed << std::setprecision(16) << eigen_solver.eigenvalues() << " "<< std::fixed<<std::endl;

    myfile2.close();

    std::ofstream myfile3 ("computed_eigenvectors_euler_play_around.gpl" , std::ios::trunc);
    std::cout << std::setprecision(16) << std::fixed;
    myfile3<< std::fixed << std::setprecision(16) << eigen_solver.eigenvectors()<< std::fixed << std::endl<<std::endl;
    myfile3.close();

    //end eigenvalues


    SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);

#endif

        const unsigned int n_global_active_cells2 = grid->n_global_active_cells();
        double delta_x = (right-left)/sqrt(n_global_active_cells2)/(poly_degree+1.0);
        pcout<<" delta x "<<delta_x<<std::endl;
         
        all_parameters_new.ode_solver_param.initial_time_step =  get_timestep(dg,poly_degree,delta_x);
         
        std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
//        double finalTime = 0.0;
        const double finalTime = all_parameters_new.flow_solver_param.final_time;
      //  const double finalTime = all_parameters_new.ode_solver_param.initial_time_step;
       // const double finalTime = 100.0*all_parameters_new.ode_solver_param.initial_time_step;


        // finalTime = 0.1;//to speed things up locally in tests, doesn't need full 14seconds to verify.
        if (all_parameters_new.use_energy == true){//for split form get energy

            double dt = all_parameters_new.ode_solver_param.initial_time_step;
            // double dt = all_parameters_new.ode_solver_param.initial_time_step / 10.0;
//            finalTime=dt;
             
            std::cout << " number dofs " << dg->dof_handler.n_dofs()<<std::endl;
            std::cout << "preparing to advance solution in time" << std::endl;
             
            // Currently the only way to calculate energy at each time-step is to advance solution by dt instead of finaltime
            // this causes some issues with outputs (only one file is output, which is overwritten at each time step)
            // also the ode solver output doesn't make sense (says "iteration 1 out of 1")
            // but it works. I'll keep it for now and need to modify the output functions later to account for this.
            double initialcond_energy = compute_kinetic_energy(dg, poly_degree);
            double initialcond_energy_mpi = (dealii::Utilities::MPI::sum(initialcond_energy, mpi_communicator));
            std::cout << std::setprecision(16) << std::fixed;
            pcout << "Energy for initial condition " << initialcond_energy_mpi/(8*pow(dealii::numbers::PI,3)) << std::endl;
             
            pcout << "Energy at time " << 0 << " is " << compute_kinetic_energy(dg, poly_degree) << std::endl;
            ode_solver->current_iteration = 0;
                ode_solver->advance_solution_time(dt/10.0);
                double initial_energy = compute_kinetic_energy(dg, poly_degree);
                double initial_energy_mpi = (dealii::Utilities::MPI::sum(initial_energy, mpi_communicator));
             
            std::cout << std::setprecision(16) << std::fixed;
            pcout << "Energy at one timestep is " << initial_energy_mpi/(8*pow(dealii::numbers::PI,3)) << std::endl;
            // std::ofstream myfile ("kinetic_energy_3D_TGV_cdg_curv_grid_4x4.gpl" , std::ios::trunc);
            std::ofstream myfile (all_parameters_new.energy_file + ".gpl"  , std::ios::trunc);
             
            for (int i = 0; i < std::ceil(finalTime/dt); ++ i) {
                ode_solver->advance_solution_time(dt);
               // ode_solver->step_in_time(dt,false);
              //  ode_solver->step_in_time(dt,true);
                // double current_energy = compute_kinetic_energy(dg,poly_degree) / initial_energy;
                double current_energy = compute_kinetic_energy(dg,poly_degree);
                double current_energy_mpi = (dealii::Utilities::MPI::sum(current_energy, mpi_communicator))/initial_energy_mpi;
                std::cout << std::setprecision(16) << std::fixed;
                // pcout << "Energy at time " << i * dt << " is " << current_energy << std::endl;
                pcout << "Energy at time " << ode_solver->current_time << " is " << current_energy_mpi << std::endl;
                pcout << "Actual Energy Divided by volume at time " << ode_solver->current_time << " is " << current_energy_mpi*initial_energy_mpi/(8*pow(dealii::numbers::PI,3)) << std::endl;
                // myfile << i * dt << " " << current_energy << std::endl;
                myfile << ode_solver->current_time << " " << current_energy_mpi << std::endl;
                // if (current_energy*initial_energy - initial_energy >= 1.00)
//                if (current_energy_mpi*initial_energy_mpi - initial_energy_mpi >= 1.00)
//                {
//                  pcout << " Energy was not monotonically decreasing" << std::endl;
//                  return 1;
//                }
                all_parameters_new.ode_solver_param.initial_time_step =  get_timestep(dg,poly_degree, delta_x);
                dt = all_parameters_new.ode_solver_param.initial_time_step;
               // ode_solver->current_iteration++;
//                if(i%10000==0){
//                    const int file_number = i / all_parameters_new.ode_solver_param.output_solution_every_x_steps;
//                    dg->output_results_vtk(file_number);
//                }
            }
             
            myfile.close();
        }
        else{//do OOA
            std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_double  = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters));
           // finalTime = 0.1;//This is sufficient for verification
           // finalTime = 0.0;//This is sufficient for verification
           //     double time_step_temp =  get_timestep(dg,poly_degree, delta_x);
           //     finalTime = 2.0 * time_step_temp;

            ode_solver->current_iteration = 0;
            ode_solver->allocate_ode_system();
//            double time_step_temp =  get_timestep(dg,poly_degree, delta_x);
//            finalTime = 2.0 * time_step_temp;
//            finalTime =0.0;
//            finalTime = 0.000150405;
//            finalTime = 0.001;
//            finalTime = 0.0001;
//            finalTime = 5.0;
//            finalTime = 23.66431913239;//20 * sqrt(1.4) since sqrt(1.4) is x-velocity magnitude and 20 is the grid
//            finalTime = 20.0*sqrt(1.4)/2.0;
//            finalTime = 20.0*sqrt(1.4)/4.0;
//            finalTime = 0.03;
            while(ode_solver->current_time < finalTime){
               // double time_step =  get_timestep(dg,poly_degree, delta_x);
                const double M_infty_temp = sqrt(2.0/1.4);
                double time_step = 0.1 * delta_x / M_infty_temp;
               // double time_step = 0.05 * delta_x / M_infty_temp;
                if(ode_solver->current_iteration%all_parameters_new.ode_solver_param.print_iteration_modulo==0)
                    pcout<<"time step "<<time_step<<" current time "<<ode_solver->current_time<<std::endl;
                double dt = dealii::Utilities::MPI::min(time_step, mpi_communicator);
                ode_solver->step_in_time(dt, false);
                ode_solver->current_iteration += 1;
                const bool is_output_iteration = (ode_solver->current_iteration % all_parameters_new.ode_solver_param.output_solution_every_x_steps == 0);
                if (is_output_iteration) {
                    const int file_number = ode_solver->current_iteration / all_parameters_new.ode_solver_param.output_solution_every_x_steps;
                    dg->output_results_vtk(file_number);
                }
            }

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
           // int overintegrate = 10;
            const int overintegrate = 10;
            dealii::QGauss<dim> quad_extra(poly_degree+1+overintegrate);
            dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
            const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
            std::array<double,nstate> soln_at_q;

            double l2error = 0.0;

            // Integrate solution error and output error
           // const double pi = atan(1)*4.0;
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
#if 0
                    const double time = ode_solver->current_time;
                    const double pi = dealii::numbers::PI;
                    const double x0 = 0.0;
                    const double y0 = 0.0;
                    const double x = qpoint[0];
                    const double y = qpoint[1];
                    const double beta = 5.0;
                   // const double beta = 0.5;
                    const double gamma = 1.4;
                   // const double r_square = (x - x0 - finalTime)*(x - x0 - finalTime) + (y - y0)*(y - y0);
                    const double r_square = (x - x0 - time)*(x - x0 - time) + (y - y0)*(y - y0);
                    const double exp_val = exp(1.0 - r_square);
                    const double numerator =  0.5*(0.4)*(beta*exp_val)*(beta*exp_val);
                    const double denominator = 8.0*gamma*pi*pi;
                    const double density = pow(1.0 - numerator/denominator, 1.0/0.4);
                    const double pressure_exact = pow(density,gamma);
#endif


//#if 0
                 //from Carolyn
                 // Setting constants
                 const double t = finalTime;
  //  const double L = 10.0; // half-width of domain
    const double L  = all_parameters->flow_solver_param.grid_right_bound;
   // const double L = 5.0; // half-width of domain
    const double pi = dealii::numbers::PI;
    const double gam = 1.4;
    const double M_infty = sqrt(2.0/gam);
    const double R = 1.0;
    const double sigma = 1.0;
    const double beta = M_infty * 5.0 * sqrt(2.0)/4.0/pi * exp(1.0/2.0);
    const double alpha = pi/4.0; //rad

    // Centre of the vortex  at t
    const double x_travel = M_infty * t * cos(alpha);
    const double x0 = 0.0 + x_travel;
    const double y_travel = M_infty * t * sin(alpha);
    const double y0 = 0.0 + y_travel;
    const double x = std::fmod(qpoint[0] - x0-L, 2.0*L)+L;
    const double y = std::fmod(qpoint[1] - y0-L, 2.0*L)+L;

    const double Omega = beta * exp(-0.5/sigma/sigma* (x/R * x/R + y/R * y/R));
//    const double delta_Ux = -y/R * Omega;
//    const double delta_Uy =  x/R * Omega;
    const double delta_T  = -(gam-1.0)/2.0 * Omega * Omega;

    // Primitive
//    const double rho = pow((1 + delta_T), 1.0/(gam-1.0));
//    const double Ux = M_infty * cos(alpha) + delta_Ux;
//    const double Uy = M_infty * sin(alpha) + delta_Uy;
//    const double Uz = 0.0;
    const double pressure_exact = 1.0/gam*pow(1+delta_T, gam/(gam-1.0));
//#endif


                    const double pressure = euler_double->compute_pressure(soln_at_q);
                    l2error += pow(pressure - pressure_exact, 2) * fe_values_extra.JxW(iquad);
                   // l2error += pow(density - soln_at_q[0], 2) * fe_values_extra.JxW(iquad);
                }
            }
            const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, mpi_communicator));

            // Convergence table
            const double dx = 1.0/pow(n_dofs/nstate,(1.0/dim));
            grid_size[igrid-igrid_start] = dx;
            soln_error[igrid-igrid_start] = l2error_mpi_sum;

            pcout << " Grid size h: " << dx 
                  << " L2-soln_error: " << l2error_mpi_sum
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
        
    }//end of OOA
}

template <int dim, int nstate>
int EulerIsentropicVortex<dim, nstate>::run_test() const
{

    using real = double;

//    const double left = -10.0;
//    const double right = 10.0;
    const double left  = all_parameters->flow_solver_param.grid_left_bound;
    const double right = all_parameters->flow_solver_param.grid_right_bound;
//    const bool colorize = true;
//    const int n_refinements = 8;
   // const int n_refinements = 32;
   // const int n_refinements = 16;
   // unsigned int poly_degree = 5;
    const unsigned int poly_degree= all_parameters->flow_solver_param.poly_degree;
   // const unsigned int poly_degree = 3;
    const unsigned int grid_degree = 1;
//    const unsigned int grid_degree = poly_degree;
//    const unsigned int igrid_start = 3;
    const unsigned int igrid_start = all_parameters->flow_solver_param.number_of_grid_elements_per_dimension;
    const unsigned int n_grids = igrid_start + 2;
    std::array<double,2> grid_size;
    std::array<double,2> soln_error;


    for(unsigned int igrid = igrid_start; igrid<n_grids; igrid++){
        using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
            MPI_COMM_WORLD,
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));
        pcout<<"got triangulation "<<std::endl;
         
        dealii::GridGenerator::hyper_cube(*grid, left, right, true);
        pcout<<"generated grid "<<std::endl;
        std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::parallel::distributed::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
        dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
        if(dim>=2) dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
        if(dim>=3) dealii::GridTools::collect_periodic_faces(*grid,4,5,2,matched_pairs);
        grid->add_periodicity(matched_pairs);
        grid->refine_global(igrid);
        this->solve(grid, poly_degree, grid_degree, left, right, igrid, igrid_start, n_grids, grid_size, soln_error);
    }//end of grid loop


    return 0;
}

#if PHILIP_DIM==2
    template class EulerIsentropicVortex <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace

