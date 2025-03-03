#include <fstream>
#include "dg/dg_factory.hpp"
#include "euler_density_wave.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "physics/initial_conditions/initial_condition_function.h"
#include "mesh/grids/straight_periodic_cube.hpp"

#include <eigen/Eigen/Eigenvalues>
#include <eigen/Eigen/Dense>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
EulerDensityWave<dim, nstate>::EulerDensityWave(const Parameters::AllParameters *const parameters_input)
    : TestsBase::TestsBase(parameters_input)
{}

template<int dim, int nstate>
double EulerDensityWave<dim, nstate>::compute_entropy(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
{
    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    //We have to project the vector of entropy variables because the mass matrix has an interpolation from solution nodes built into it.
    OPERATOR::vol_projection_operator<dim,2*dim,double> vol_projection(1, poly_degree, dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    OPERATOR::basis_functions<dim,2*dim,double> soln_basis(1, poly_degree, dg->max_grid_degree);
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global(dg->right_hand_side);
    dealii::LinearAlgebra::distributed::Vector<double> energy_var_hat_global(dg->right_hand_side);
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_double  = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters));

    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);

        //get solution modal coeff
        std::array<std::vector<double>,nstate> soln_coeff;
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0)
                soln_coeff[istate].resize(n_shape_fns);
            soln_coeff[istate][ishape] = dg->solution(dofs_indices[idof]);
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
     //   std::array<std::vector<double>,nstate> energy_var_at_q;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<double,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }
            std::array<double,nstate> entropy_var_state = euler_double->compute_entropy_variables(soln_state);
        //    std::array<double,nstate> kin_energy_state = euler_double->compute_kinetic_energy_variables(soln_state);
            for(int istate=0; istate<nstate; istate++){
                if(iquad==0){
                    entropy_var_at_q[istate].resize(n_quad_pts);
        //            energy_var_at_q[istate].resize(n_quad_pts);
                }
        //        energy_var_at_q[istate][iquad] = kin_energy_state[istate];
                entropy_var_at_q[istate][iquad] = entropy_var_state[istate];
            }
        }
        //project the enrtopy and KE var to modal coefficients
        //then write it into a global vector
        for(int istate=0; istate<nstate; istate++){
            //Projected vector of entropy variables.
            std::vector<double> entropy_var_hat(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(entropy_var_at_q[istate], entropy_var_hat,
                                                 vol_projection.oneD_vol_operator);
        //    std::vector<double> energy_var_hat(n_shape_fns);
        //    vol_projection.matrix_vector_mult_1D(energy_var_at_q[istate], energy_var_hat,
        //                                         vol_projection.oneD_vol_operator);

            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                const unsigned int idof = istate * n_shape_fns + ishape;
                entropy_var_hat_global[dofs_indices[idof]] = entropy_var_hat[ishape];
       //         energy_var_hat_global[dofs_indices[idof]] = energy_var_hat[ishape];
            }
        }
    }

    //evaluate the change in entropy and change in KE
    dg->assemble_residual();
    double change_entropy_and_energy;
    change_entropy_and_energy = entropy_var_hat_global * dg->right_hand_side;
    return change_entropy_and_energy;

}

template<int dim, int nstate>
double EulerDensityWave<dim, nstate>::compute_kinetic_energy(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
{
    //returns the energy in the L2-norm (physically relevant)
    int overintegrate = 0 ;
    dealii::QGauss<dim> quad_extra(dg->max_degree+1+overintegrate);
    const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::FEValues<dim,dim> fe_values_extra(mapping, dg->fe_collection[poly_degree], quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::array<double,nstate> soln_at_q;

    double total_kinetic_energy = 0;

    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_double  = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters));

    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;

        fe_values_extra.reinit (cell);
        dofs_indices.resize(fe_values_extra.dofs_per_cell);
        cell->get_dof_indices (dofs_indices);

        //Please see Eq. 3.21 in Gassner, Gregor J., Andrew R. Winters, and David A. Kopriva. "Split form nodal discontinuous Galerkin schemes with summation-by-parts property for the compressible Euler equations." Journal of Computational Physics 327 (2016): 39-66.
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
            for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
             const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
            }

           // const double density = soln_at_q[0];

           // double quadrature_kinetic_energy = 0.0;
           // for(int i=0;i<dim;i++){
           //     quadrature_kinetic_energy += 0.5*(soln_at_q[i+1]*soln_at_q[i+1])/density;
           // }
           // total_kinetic_energy += quadrature_kinetic_energy * fe_values_extra.JxW(iquad);
            const double density = soln_at_q[0];
            const double pressure = euler_double->compute_pressure(soln_at_q);
            const double entropy = log(pressure) - euler_double->gam * log(density);
            const double quadrature_entropy = -density*entropy/euler_double->gamm1;
            
            total_kinetic_energy += quadrature_entropy * fe_values_extra.JxW(iquad);
           // total_kinetic_energy += quadrature_kinetic_energy;

//            dealii::Tensor<1,dim,double> vel ;
//            for(int idim=0;idim<dim; idim++){
//                vel[idim] = soln_at_q[idim+1]/soln_at_q[0];
//            }
//            double pressure = 0.4*soln_at_q[nstate-1];
//            for(int idim=0; idim<dim; idim++){
//                pressure -= 0.5*soln_at_q[0]*vel[idim]*vel[idim];
//            }
//            const double s = log(pressure) - 1.4*log(density);
//            total_kinetic_energy -= density*s/0.4;
//           // total_kinetic_energy -= density*s/0.4 * fe_values_extra.JxW(iquad);

        }
    }
    return total_kinetic_energy;
}

template<int dim, int nstate>
double EulerDensityWave<dim, nstate>::compute_conservation(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
{
        //Conservation \f$ =  \int 1 * u d\Omega_m \f$
        double conservation = 0.0;
        dealii::LinearAlgebra::distributed::Vector<double> mass_matrix_times_solution(dg->right_hand_side);
        if(dg->all_parameters->use_inverse_mass_on_the_fly)
            dg->apply_global_mass_matrix(dg->solution,mass_matrix_times_solution);
        else
            dg->global_mass_matrix.vmult( mass_matrix_times_solution, dg->solution);

        const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
        const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
        std::vector<double> ones(n_quad_pts, 1.0);
        // std::vector<double> ones(n_quad_pts);
        // for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
        //     ones[iquad] = 1.0;
        // }
        // Projected vector of ones. That is, the interpolation of ones_hat to the volume nodes is 1.
        std::vector<double> ones_hat(n_quad_pts);
        // We have to project the vector of ones because the mass matrix has an interpolation from solution nodes built into it.
        OPERATOR::vol_projection_operator<dim,2*dim,double> vol_projection(1, poly_degree, dg->max_grid_degree);
        vol_projection.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);
        vol_projection.matrix_vector_mult_1D(ones, ones_hat,
                                                   vol_projection.oneD_vol_operator);

        dealii::LinearAlgebra::distributed::Vector<double> ones_hat_global(dg->right_hand_side);
        std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);
        for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
            if (!cell->is_locally_owned()) continue;
            cell->get_dof_indices (dofs_indices);
            for(int istate=0; istate<nstate; istate++){
                for(unsigned int ishape=0;ishape<n_quad_pts; ishape++){
                    const unsigned int idof = istate*n_quad_pts + ishape;
                    ones_hat_global[dofs_indices[idof]] = ones_hat[ishape];
                }
            }
        }

        conservation = ones_hat_global * mass_matrix_times_solution;

    return conservation;
}

template<int dim, int nstate>
double EulerDensityWave<dim, nstate>::get_timestep(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree, const double delta_x) const
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
       // double cfl = 0.05 * delta_x/max_eig;
       // double cfl = 0.1 * delta_x/max_eig;
       // double cfl = 0.2 * delta_x/max_eig;
        //double cfl = 0.5 * delta_x/max_eig;
        double cfl = delta_x/max_eig * dg->all_parameters->flow_solver_param.courant_friedrichs_lewy_number;

       // double cfl = 0.000005 * delta_x/max_eig;
        if(cfl < cfl_min)
            cfl_min = cfl;
    }
    return cfl_min;
}

template <int dim, int nstate>
int EulerDensityWave<dim, nstate>::run_test() const
{
//#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
//        using Triangulation = dealii::Triangulation<dim>;
//        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
//            typename dealii::Triangulation<dim>::MeshSmoothing(
//                dealii::Triangulation<dim>::smoothing_on_refinement |
//                dealii::Triangulation<dim>::smoothing_on_coarsening));
//#else
//        using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
//        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
//            MPI_COMM_WORLD,
//            typename dealii::Triangulation<dim>::MeshSmoothing(
//                dealii::Triangulation<dim>::smoothing_on_refinement |
//                dealii::Triangulation<dim>::smoothing_on_coarsening));
//#endif

    using real = double;

    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;  
    const double left = all_parameters_new.flow_solver_param.grid_left_bound;
    const double right = all_parameters_new.flow_solver_param.grid_right_bound;
    const unsigned int grid_degree = all_parameters_new.flow_solver_param.grid_degree;
    const unsigned int num_grid_elem_per_dir = all_parameters_new.flow_solver_param.number_of_grid_elements_per_dimension;
    const unsigned int poly_degree = all_parameters_new.flow_solver_param.poly_degree;
//    double left = -1.0;
//    double right = 1.0;
////    const bool colorize = true;
//   // const int n_refinements = 4;
//    const int n_refinements = 2;
//   // const int n_refinements = 8;
////    const int n_refinements = 32;
//  //  const int n_refinements = 16;
//    const unsigned int poly_degree = 5;
//  //  unsigned int poly_degree = 3;
//   // unsigned int poly_degree = 8;
//    const unsigned int grid_degree = 1;
////    const unsigned int grid_degree = poly_degree;

    // set the warped grid
   // PHiLiP::Grids::nonsymmetric_curved_grid<dim,Triangulation>(*grid, n_refinements);

//    PHiLiP::Grids::straight_periodic_cube<dim,Triangulation>(grid, left, right, n_refinements);
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
   //  dealii::GridGenerator::hyper_cube(*grid, left, right, true);
   //  std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
   //  dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
   //  if constexpr (dim>1)
   //     dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
   //  if constexpr (dim>2)
   //     dealii::GridTools::collect_periodic_faces(*grid,4,5,2,matched_pairs);
   //  grid->add_periodicity(matched_pairs);
   //  grid->refine_global(n_refinements);

    // Create DG
    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
    dg->allocate_system ();

    std::cout << "Implement initial conditions" << std::endl;
    std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function = 
                InitialConditionFactory<dim,nstate,double>::create_InitialConditionFunction(&all_parameters_new);
    SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);


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
  //  const double perturbation = 1e-5;
    std::cout<<"doing perturbations"<<std::endl;
    for(unsigned int eig_direction=0; eig_direction<dg->solution.size(); eig_direction++){
        double solution_init_value = dg->solution[eig_direction];
        dg->solution[eig_direction] += perturbation;
        dg->assemble_residual();

//    SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);

        if(all_parameters_new.use_inverse_mass_on_the_fly){
            dg->apply_inverse_global_mass_matrix(dg->right_hand_side, col_dRdU); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
        } else{
            dg->global_inverse_mass_matrix.vmult(col_dRdU, dg->right_hand_side); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
        }
        for(unsigned int i=0; i<dg->solution.size(); i++){
           // dRdU[i][eig_direction] = dg->right_hand_side[i];
           //col_dRdU[i] = dg->right_hand_side[i];
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
           // dRdU[i][eig_direction] -= dg->right_hand_side[i];
           //col_dRdU[i] = dg->right_hand_side[i];
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
        for(unsigned int i=0; i<eigen_solver.eigenvalues().size(); i++){
            if(eigen_solver.eigenvalues()[i].real() > 1e-6){
                pcout<<" eigenvalue "<<eigen_solver.eigenvalues()[i].real()<<std::endl;
            }
        }


   // eigen_solver.compute(dRdU_mpi);
    std::cout<<"got eigs"<<std::endl;
    std::ofstream myfile2 ("computed_eigenvalues_euler_LIA_flux1.gpl" , std::ios::trunc);
    std::cout << std::setprecision(16) << std::fixed;
    myfile2<< std::fixed << std::setprecision(16) << eigen_solver.eigenvalues() << " "<< std::fixed<<std::endl;

    myfile2.close();

    std::ofstream myfile3 ("computed_eigenvectors_euler_LIA_flux1.gpl" , std::ios::trunc);
    std::cout << std::setprecision(16) << std::fixed;
    myfile3<< std::fixed << std::setprecision(16) << eigen_solver.eigenvectors()<< std::fixed << std::endl<<std::endl;
    myfile3.close();

    //end eigenvalues
std::cout<<"did eigenvalues"<<std::endl;
std::abort();
#endif


    SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);


    const unsigned int n_global_active_cells2 = grid->n_global_active_cells();
    double delta_x = (right-left)/pow(n_global_active_cells2,1.0/dim)/(poly_degree+1.0);
    pcout<<" delta x "<<delta_x<<std::endl;

    all_parameters_new.ode_solver_param.initial_time_step =  get_timestep(dg,poly_degree,delta_x);
     
    std::cout << "creating ODE solver" << std::endl;
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    std::cout << "ODE solver successfully created" << std::endl;
   // double finalTime = 5.;
    double finalTime = all_parameters_new.flow_solver_param.final_time;
    std::cout<<"Final time "<<finalTime<<std::endl;
    finalTime = 15.0;
    // finalTime = 0.1;//to speed things up locally in tests, doesn't need full 14seconds to verify.
    double dt = all_parameters_new.ode_solver_param.initial_time_step;
    // double dt = all_parameters_new.ode_solver_param.initial_time_step / 10.0;
   // finalTime=dt;
//    finalTime=2.0;
  // finalTime = 100.0*dt;
 // finalTime =0.0;

//finalTime=10.0;
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
//	ode_solver->advance_solution_time(dt/10.0);
	double initial_energy = compute_kinetic_energy(dg, poly_degree);
	double initial_energy_mpi = (dealii::Utilities::MPI::sum(initial_energy, mpi_communicator));
        double initial_conservation = compute_conservation(dg, poly_degree);
	double initial_conservation_mpi = (dealii::Utilities::MPI::sum(initial_conservation, mpi_communicator));
//	const double initial_entropy = compute_entropy(dg, poly_degree);
//	const double initial_entropy_mpi = (dealii::Utilities::MPI::sum(initial_entropy, mpi_communicator));

    std::cout << std::setprecision(16) << std::fixed;
    pcout << "Energy at one timestep is " << initial_energy_mpi/(8*pow(dealii::numbers::PI,3)) << std::endl;
    // std::ofstream myfile ("kinetic_energy_3D_TGV_cdg_curv_grid_4x4.gpl" , std::ios::trunc);
    std::ofstream myfile (all_parameters_new.energy_file + ".gpl"  , std::ios::trunc);
   // std::ofstream myfile2 ("solution_vs_eaxct.gpl"  , std::ios::trunc);

//    for (int i = 0; i < std::ceil(finalTime/dt); ++ i) {
   // while(ode_solver->current_time <= finalTime){
    ode_solver->current_iteration = 0;
    ode_solver->allocate_ode_system();
    const int file_number = ode_solver->current_iteration / all_parameters_new.ode_solver_param.output_solution_every_x_steps;
    dg->output_results_vtk(file_number);

    //set min entropy for limiter
  //  ode_solver->limiter->set_cell_min_entropy(dg->solution,dg->dof_handler,dg->fe_collection,dg->volume_quadrature_collection,1,poly_degree,dg->oneD_fe_collection_1state, dg->oneD_quadrature_collection);
#if 0
    //write to file the first
        dealii::QGaussLobatto<dim> quad_extra(poly_degree+1);
        dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], quad_extra, 
                dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
        const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
        std::array<double,nstate> soln_at_q;
        std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
        std::cout << std::setprecision(16) << std::fixed;
        myfile2<<ode_solver->current_time<<std::endl;
        //write x
        for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
            if (!cell->is_locally_owned()) continue;
        
            fe_values_extra.reinit (cell);
            cell->get_dof_indices (dofs_indices);
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));

                std::cout << std::setprecision(16) << std::fixed;
                myfile2<< std::fixed << std::setprecision(16) << qpoint[0] <<std::endl;
            }
        }
        //write approx
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
                    std::cout << std::setprecision(16) << std::fixed;
                    myfile2 << std::fixed << std::setprecision(16) <<" " << soln_at_q[0]<< std::endl;
                }
            }
            //write exact
            for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
                if (!cell->is_locally_owned()) continue;
            
                fe_values_extra.reinit (cell);
                cell->get_dof_indices (dofs_indices);
                double pi = dealii::numbers::PI;
                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));

                    std::cout << std::setprecision(16) << std::fixed;
                    myfile2<< std::fixed << std::setprecision(16) << " " << 1.0+0.98*sin(pi*(qpoint[0]-0.1*ode_solver->current_time))<< std::endl;
                }
            }
#endif

    while(ode_solver->current_time < finalTime){
//        ode_solver->advance_solution_time(dt);
        all_parameters_new.ode_solver_param.initial_time_step =  get_timestep(dg,poly_degree, delta_x);
        dt = all_parameters_new.ode_solver_param.initial_time_step;
        ode_solver->step_in_time(dt,false);
      //  ode_solver->step_in_time(dt,true);
        // double current_energy = compute_kinetic_energy(dg,poly_degree) / initial_energy;
        const bool is_output_iteration = (ode_solver->current_iteration % all_parameters_new.ode_solver_param.output_solution_every_x_steps == 0);
        if (is_output_iteration) {
            double current_energy = compute_kinetic_energy(dg,poly_degree);
            double current_energy_mpi = (dealii::Utilities::MPI::sum(current_energy, mpi_communicator))/initial_energy_mpi;
            std::cout << std::setprecision(16) << std::fixed;
            // pcout << "Energy at time " << i * dt << " is " << current_energy << std::endl;
           // pcout << "Energy at time " << i * dt << " is " << current_energy_mpi << std::endl;
            pcout << "Energy at time " << ode_solver->current_time << " is " << current_energy_mpi << std::endl;
            pcout << "Actual Energy Divided by volume at time " << ode_solver->current_time << " is " << current_energy_mpi*initial_energy_mpi/(8*pow(dealii::numbers::PI,3)) << std::endl;
            // myfile << i * dt << " " << current_energy << std::endl;
            myfile << ode_solver->current_time << " " << current_energy_mpi << std::endl;
            // if (current_energy*initial_energy - initial_energy >= 1.00)
//            if (current_energy_mpi*initial_energy_mpi - initial_energy_mpi >= 1.00)
//            {
//              pcout << " Energy was not monotonically decreasing" << std::endl;
//              return 1;
//            }
            double current_conservation = compute_conservation(dg, poly_degree);
            double current_conservation_mpi = (dealii::Utilities::MPI::sum(current_conservation, mpi_communicator))/initial_conservation_mpi;
            std::cout << std::setprecision(16) << std::fixed;
            this->pcout << "Normalized Conservation at time " << ode_solver->current_time << " is " << current_conservation_mpi<< std::endl;
             
            double current_entropy = compute_entropy(dg,poly_degree);
           // double current_entropy_mpi = (dealii::Utilities::MPI::sum(current_entropy, mpi_communicator))/initial_entropy_mpi;
            double current_entropy_mpi = (dealii::Utilities::MPI::sum(current_entropy, mpi_communicator));
            pcout << "Entropy change at time " << ode_solver->current_time << " is " << current_entropy_mpi << std::endl;
             
             
           // ode_solver->current_iteration += 1;


            const int file_number = ode_solver->current_iteration / all_parameters_new.ode_solver_param.output_solution_every_x_steps;
            dg->output_results_vtk(file_number);
        }
    }

    std::cout<<"Done time integration"<<std::endl;
#if 0
        //output soln vs exact
        if (is_output_iteration) {
            std::cout << std::setprecision(16) << std::fixed;
            myfile2<<ode_solver->current_time<<std::endl;
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
                    std::cout << std::setprecision(16) << std::fixed;
                    myfile2 << std::fixed << std::setprecision(16) <<" " << soln_at_q[0]<< std::endl;
                   // double pi = dealii::numbers::PI;
                   // const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));
                   // double exact = 1.0+0.98*sin(pi*(qpoint[0]-0.1*ode_solver->current_time));
                   // pcout<<"Difference between approx and exact "<<abs(exact - soln_at_q[0])<<std::endl;
                }
            }
            for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
                if (!cell->is_locally_owned()) continue;
            
                fe_values_extra.reinit (cell);
                cell->get_dof_indices (dofs_indices);
                double pi = dealii::numbers::PI;
                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));

                    std::cout << std::setprecision(16) << std::fixed;
                    myfile2<< std::fixed << std::setprecision(16) << " " << 1.0+0.98*sin(pi*(qpoint[0]-0.1*ode_solver->current_time))<< std::endl;
                }
            }
        }
#endif


//        if(i%10000==0){
//            const int file_number = i / all_parameters_new.ode_solver_param.output_solution_every_x_steps;
//            dg->output_results_vtk(file_number);
//        }

        
//    std::vector<dealii::types::global_dof_index> dofs_indices1 (3*(poly_degree+1));
//       for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
//       if (!cell->is_locally_owned()) continue;
//
//       cell->get_dof_indices (dofs_indices1);
//          // for(unsigned int i=0;i<=poly_degree;i++){
//           for(unsigned int i=0;i<3*(poly_degree+1);i++){
//           pcout<<" rish hand side "<<dg->right_hand_side[dofs_indices1[i]]<<" for idof "<<i<<" with global dof "<<dofs_indices1[i]<<std::endl;
//         //  pcout<<" density "<<dg->solution[dofs_indices1[i]]<<" for idof "<<i<<" with global dof "<<dofs_indices1[i]<<std::endl;
//           }
//       }

   // }

    myfile.close();
   // myfile2.close();

    return 0;
}

template class EulerDensityWave <PHILIP_DIM,PHILIP_DIM+2>;

} // Tests namespace
} // PHiLiP namespace



