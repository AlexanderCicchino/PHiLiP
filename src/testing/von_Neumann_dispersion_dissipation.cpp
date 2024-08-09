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

#include "von_Neumann_dispersion_dissipation.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "dg/dg_base.hpp"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_base.h"
#include <fstream>
#include "ode_solver/ode_solver_factory.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "physics/initial_conditions/initial_condition_function.h"
#include "mesh/grids/straight_periodic_cube.hpp"

#include <eigen/Eigen/Eigenvalues>
#include <eigen/Eigen/Dense>
#include "linear_solver/helper_functions.cpp"
#include <complex>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
VonNeumannDispersionDissipation<dim, nstate>::VonNeumannDispersionDissipation(const PHiLiP::Parameters::AllParameters *const parameters_input)
: TestsBase::TestsBase(parameters_input)
{}

template<int dim, int nstate>
std::array<std::vector<double>,nstate> VonNeumannDispersionDissipation<dim, nstate>::compute_u_avg(
    const std::shared_ptr < DGBase<dim, double> > &dg, 
    const unsigned int poly_degree, 
    const unsigned int n_elem, 
    const double /*left*/, const double /*right*/) const
{
    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    const unsigned int grid_degree = dg->high_order_grid->fe_system.tensor_degree();

    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, dg->max_grid_degree);
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    OPERATOR::mapping_shape_functions<dim,2*dim> mapping_basis(1, poly_degree, grid_degree);
    mapping_basis.build_1D_shape_functions_at_grid_nodes(dg->high_order_grid->oneD_fe_system, dg->high_order_grid->oneD_grid_nodes);
    mapping_basis.build_1D_shape_functions_at_flux_nodes(dg->high_order_grid->oneD_fe_system, dg->oneD_quadrature_collection[poly_degree], dg->oneD_face_quadrature);
    const std::vector<double> &vol_quad_weights = dg->volume_quadrature_collection[poly_degree].get_weights();

    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

    std::array<std::vector<double>,nstate> avg_soln;
    for(int istate=0; istate<nstate; istate++){
        avg_soln[istate].resize(n_elem);
    }
//    const double delx = (right-left)/n_elem;
    auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
    unsigned int ielem = 0;
    for (auto cell = dg->dof_handler.begin_active(); cell!= dg->dof_handler.end(); ++cell, ++metric_cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);

        // We first need to extract the mapping support points (grid nodes) from high_order_grid.
        const dealii::FESystem<dim> &fe_metric = dg->high_order_grid->fe_system;
        const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
        const unsigned int n_grid_nodes  = n_metric_dofs / dim;
        std::vector<dealii::types::global_dof_index> metric_dof_indices(n_metric_dofs);
        metric_cell->get_dof_indices (metric_dof_indices);
        std::array<std::vector<double>,dim> mapping_support_points;
        for(int idim=0; idim<dim; idim++){
            mapping_support_points[idim].resize(n_grid_nodes);
        }
        // Get the mapping support points (physical grid nodes) from high_order_grid.
        // Store it in such a way we can use sum-factorization on it with the mapping basis functions.
        const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(grid_degree);
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const double val = (dg->high_order_grid->volume_nodes[metric_dof_indices[idof]]);
            const unsigned int istate = fe_metric.system_to_component_index(idof).first;
            const unsigned int ishape = fe_metric.system_to_component_index(idof).second;
            const unsigned int igrid_node = index_renumbering[ishape];
            mapping_support_points[istate][igrid_node] = val;
        }
        // Construct the metric operators.
        OPERATOR::metric_operators<double, dim, 2*dim> metric_oper(nstate, poly_degree, grid_degree, false, false);
        // Build the metric terms to compute the gradient and volume node positions.
        // This functions will compute the determinant of the metric Jacobian and metric cofactor matrix.
        // If flags store_vol_flux_nodes and store_surf_flux_nodes set as true it will also compute the physical quadrature positions.
        metric_oper.build_volume_metric_operators(
            n_quad_pts, n_grid_nodes,
            mapping_support_points,
            mapping_basis,
            dg->all_parameters->use_invariant_curl_form);



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
            // Interpolate soln coeff to volume cubature nodes.
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }
        for(int istate=0; istate<nstate; istate++){
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
               // avg_soln[istate][ielem] += soln_at_q[istate][iquad] * vol_quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
                avg_soln[istate][ielem] += soln_at_q[istate][iquad] * vol_quad_weights[iquad];
               // avg_soln[istate][ielem] += soln_at_q[istate][iquad];
            }
        //    avg_soln[istate][ielem] /= delx;
           // avg_soln[istate][ielem] /= n_quad_pts;
        }

        ielem++;
    }

    return avg_soln;
}
template <int dim, int nstate>
int VonNeumannDispersionDissipation<dim, nstate>::run_test() const
{
    pcout << " Running von Neumann stability analysis for dispersion and dissipation. " << std::endl;

    //Initial parameters
    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;  
    const double left = all_parameters_new.flow_solver_param.grid_left_bound;
    const double right = all_parameters_new.flow_solver_param.grid_right_bound;
    const unsigned int grid_degree = all_parameters_new.flow_solver_param.grid_degree;
    const unsigned int num_grid_elem_per_dir = all_parameters_new.flow_solver_param.number_of_grid_elements_per_dimension;
    const unsigned int poly_degree = all_parameters_new.flow_solver_param.poly_degree;
    const unsigned int n_dof = poly_degree + 1;
    const unsigned int n_face_non_per = num_grid_elem_per_dir - 1;
    const double delx = (right-left)/num_grid_elem_per_dir;

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

    //Initialize DG
    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
    dg->allocate_system (true,false,false);

    //Implement initial condition
    std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function = 
                InitialConditionFactory<dim,nstate,double>::create_InitialConditionFunction(&all_parameters_new);
    SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);

    const std::array<std::vector<double>,nstate> soln_avg = compute_u_avg(dg, poly_degree, num_grid_elem_per_dir, left, right);

    std::vector<double> elem_wave_speed(num_grid_elem_per_dir);
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    PDE_enum pde_type = all_parameters_new.pde_type;

    double avg_wave_speed=0.0;
    for(unsigned int ielem=0; ielem<num_grid_elem_per_dir; ielem++){
        if(pde_type == PDE_enum::advection){
            elem_wave_speed[ielem] = 1.1;
            avg_wave_speed = 1.1;
        }
        if(pde_type == PDE_enum::burgers_inviscid){
            elem_wave_speed[ielem] = soln_avg[0][ielem];
            std::cout<<"elem_wave_speed "<<elem_wave_speed[ielem]<<" for elem "<<ielem<<std::endl;
            avg_wave_speed+= elem_wave_speed[ielem];
        }
    }
    std::cout<<"avg "<<avg_wave_speed<<std::endl;

    for(unsigned int ielem=0; ielem<num_grid_elem_per_dir; ielem++){
        std::cout<<"element "<<ielem<<std::endl;
        for(unsigned int idof=0; idof<n_dof; idof++){
            std::cout<<"solution "<<dg->solution[ielem*n_dof + idof]<<" for idof "<<idof<<std::endl;
        }
    }


    
    //to compute eigenvalues/eigenvectors
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eigen_solver;


    //Build the residual's Jacobian
    Eigen::MatrixXcd dRdU(dg->solution.size(), dg->solution.size());
    dg->assemble_residual(true);
    dg->evaluate_mass_matrices(true);
    dealii::TrilinosWrappers::SparseMatrix mass_inv_dRdU;
    mass_inv_dRdU.reinit(dg->locally_owned_dofs, dg->sparsity_pattern, MPI_COMM_WORLD);
    dg->global_inverse_mass_matrix.mmult(mass_inv_dRdU, dg->system_matrix);

    //write system matrix to dRdU
    for(unsigned int idof_global=0; idof_global<dg->solution.size(); idof_global++){
        for(unsigned int jdof_global=0; jdof_global<dg->solution.size(); jdof_global++){
            dRdU(idof_global,jdof_global) = mass_inv_dRdU(idof_global,jdof_global);
        }
    }

//#if 0
    //Legendre polynomial
    const dealii::FE_DGQLegendre<1> fe_dg(poly_degree);
    const dealii::FESystem<1,1> fe_system(fe_dg, 1);
    OPERATOR::basis_functions<dim,2*dim> legendre_poly(1, poly_degree, grid_degree); 
    legendre_poly.build_1D_volume_operator(fe_system,dg->oneD_quadrature_collection[poly_degree]);
    Eigen::MatrixXcd legendre_mat(n_dof, n_dof);
    Eigen::MatrixXcd legendre_mat_inv(n_dof, n_dof);
    for(unsigned int idof=0; idof<n_dof; idof++){
        for(unsigned int jdof=0; jdof<n_dof; jdof++){
            legendre_mat(idof,jdof) = legendre_poly.oneD_vol_operator[idof][jdof];
        }
    }
    legendre_mat_inv = legendre_mat.inverse();
//    std::cout<<"legendre poly"<<std::endl;
//    for(unsigned int i=0; i<poly_degree+1;i++){
//    for(unsigned int j=0; j<poly_degree+1;j++){
//        std::cout<<legendre_poly.oneD_vol_operator[i][j]<<" ";
//    }
//    std::cout<<std::endl;
//    }

//#endif

    std::vector<std::complex<double>> right_ghost_BC(n_dof);
    std::vector<std::complex<double>> left_ghost_BC(n_dof);
    std::vector<std::complex<double>> first_elem(n_dof);
    std::vector<std::complex<double>> last_elem(n_dof);
    for(unsigned int idof=0; idof<n_dof; idof++){
        left_ghost_BC[idof] = dRdU(idof, dg->solution.size()-1);
        right_ghost_BC[idof] = dRdU(dg->solution.size() - n_dof + idof, 0);
        dRdU(idof, dg->solution.size()-1) = 0.0;
        dRdU(dg->solution.size() - n_dof + idof, 0) = 0.0;
        first_elem[idof] = dRdU(idof,n_dof-1);//last column of first elem
        last_elem[idof] = dRdU(dg->solution.size() - n_dof + idof,dg->solution.size()-n_dof);//first column of last elem
    }
    std::vector<std::complex<double>> right_face(n_face_non_per*n_dof);
    std::vector<std::complex<double>> left_face(n_face_non_per*n_dof);
    std::vector<std::complex<double>> int_right_face(n_face_non_per*n_dof);
    std::vector<std::complex<double>> int_left_face(n_face_non_per*n_dof);
    for(unsigned int i=0; i<n_face_non_per;i++){
        const unsigned int right_index_col = (i+1)*(n_dof);
        const unsigned int right_index_row = (i)*(n_dof);
        const unsigned int left_index_col = (i)*(n_dof)+n_dof-1;
        const unsigned int left_index_row = (i+1)*(n_dof);
        for(unsigned int idof=0; idof<n_dof; idof++){
            right_face[i*n_dof + idof] = dRdU(right_index_row+idof, right_index_col);
            left_face[i*n_dof + idof]  = dRdU(left_index_row+idof, left_index_col);
            //zero out the term since brought over to local solution
            dRdU(right_index_row+idof, right_index_col) = 0.0;
            dRdU(left_index_row+idof, left_index_col) = 0.0;
             
            int_right_face[i*n_dof + idof] = dRdU(right_index_row+ idof, right_index_col - n_dof);//first column of interior elem
            int_left_face[i*n_dof + idof]  = dRdU(left_index_row+idof, left_index_col + n_dof);//last column of interior elem
        }
    }

    //Apply Fourier mode to the Jacobian
    const std::complex<double> imag(0.0,1.0);
//    std::ofstream myfile ("nonlinear_von_Neumann_eigenvalues_dispersive.gpl" , std::ios::trunc);
//    std::ofstream myfile2 ("nonlinear_von_Neumann_eigenvalues_dissipative.gpl" , std::ios::trunc);
//    std::ofstream myfile3 ("nonlinear_von_Neumann_eigenvectors.gpl" , std::ios::trunc);
    //std::ofstream myfile4 ("nonlinear_von_Neumann_coeff_times_eigenvectors.gpl" , std::ios::trunc);
    std::ofstream myfile5 ("nonlinear_von_Neumann_eigenvalues_local.gpl" , std::ios::trunc);
    std::ofstream myfile6 ("nonlinear_von_Neumann_eigenvalues_local_physical.gpl" , std::ios::trunc);
    double pi = dealii::numbers::PI;
    double slope_prev = 0.0;
    bool start_flag = false;
    double eig_prev = 0.0;
    double eig_prev_diss = 0.0;
    double slope_prev_diss = 0.0;
//    bool jump_flag = false;
    for(double fourier_mode= -1.0 *(poly_degree+1)*pi; fourier_mode < (poly_degree+1)*pi; fourier_mode+=0.1){

        //Impose BC Fourier Modes
        for(unsigned int idof=0; idof<n_dof; idof++){
            dRdU(idof,n_dof-1) = first_elem[idof] + left_ghost_BC[idof] * exp(-imag * fourier_mode);//del_x_ref = 1
            dRdU(dg->solution.size() - n_dof + idof,dg->solution.size()-n_dof) = last_elem[idof] + right_ghost_BC[idof] * exp(imag * fourier_mode);//del_x_ref = 1
        }
        //Impose Fourier Modes interior faces
        for(unsigned int i=0; i<n_face_non_per;i++){
            const unsigned int right_index_col = (i+1)*(n_dof);
            const unsigned int right_index_row = (i)*(n_dof);
            const unsigned int left_index_col = (i)*(n_dof) + n_dof - 1;
            const unsigned int left_index_row = (i+1)*(n_dof);
            for(unsigned int idof=0; idof<n_dof; idof++){
                dRdU(right_index_row+ idof, right_index_col - n_dof) = int_right_face[i*n_dof + idof] + right_face[i *n_dof + idof] * exp(imag * fourier_mode);
                dRdU(left_index_row+ idof, left_index_col + n_dof) = int_left_face[i*n_dof + idof] + left_face[i *n_dof + idof] * exp(-imag * fourier_mode);
            }
        }

    #if 0
        //Compute the eigenvalues and eigenvectors of the Jacobian with Fourier mode
        eigen_solver.compute(dRdU);
         
        //Print to a file the eigenvalues vs x to plot
        myfile<<fourier_mode<<std::endl;
       // myfile<< std::fixed << std::setprecision(16) << - eigen_solver.eigenvalues().imag()/avg_wave_speed << " "<< std::fixed<<std::endl;
        myfile<< std::fixed << std::setprecision(16) << - eigen_solver.eigenvalues().imag() << " "<< std::fixed<<std::endl;
         
        myfile2<<fourier_mode<<std::endl;
       // myfile2<< std::fixed << std::setprecision(16) << eigen_solver.eigenvalues().real()/avg_wave_speed << " "<< std::fixed<<std::endl;
        myfile2<< std::fixed << std::setprecision(16) << eigen_solver.eigenvalues().real() << " "<< std::fixed<<std::endl;

        myfile3<<fourier_mode<<std::endl;
        myfile3<< std::fixed << std::setprecision(16) << eigen_solver.eigenvectors()<< std::fixed << std::endl<<std::endl;

        //Extract coefficients for eigenvector basis
        Eigen::MatrixXcd eigenvectors(dg->solution.size(), dg->solution.size());
        Eigen::MatrixXcd eigenvectors_inv(dg->solution.size(), dg->solution.size());
        eigenvectors = eigen_solver.eigenvectors();
        eigenvectors_inv = eigenvectors.inverse();
        Eigen::VectorXcd eigenvector_coeff(dg->solution.size());
        for(unsigned int i=0; i<dg->solution.size(); i++){
            eigenvector_coeff[i] = 0.0;
            for(unsigned int j=0; j<dg->solution.size(); j++){
                eigenvector_coeff[i] += eigenvectors_inv(i,j) * dg->solution[j];
            }
        }
    #endif

        //Extract local Jacobian and compute local eigenvalues
        Eigen::MatrixXcd dRdU_local(n_dof, n_dof);
        myfile5<<fourier_mode<<std::endl;
        for(unsigned int ielem=0; ielem<num_grid_elem_per_dir; ielem++){
            for(unsigned int idof=0; idof<n_dof; idof++){
                for(unsigned int jdof=0; jdof<n_dof; jdof++){
                    dRdU_local(idof,jdof) = dRdU(ielem*n_dof + idof, ielem*n_dof + jdof);
                }
            }
            eigen_solver.compute(dRdU_local);
            myfile5<<ielem<<std::endl;
            myfile5<< std::fixed << std::setprecision(16) << - eigen_solver.eigenvalues().imag()/elem_wave_speed[ielem] << " "<< std::fixed<<std::endl;
            myfile5<< std::fixed << std::setprecision(16) << eigen_solver.eigenvalues().real()/elem_wave_speed[ielem] << " "<< std::fixed<<std::endl;
         //   myfile5<< std::fixed << std::setprecision(16) << - eigen_solver.eigenvalues().imag() << " "<< std::fixed<<std::endl;
         //   myfile5<< std::fixed << std::setprecision(16) << eigen_solver.eigenvalues().real() << " "<< std::fixed<<std::endl;

            //Extract physical mode
            if(fourier_mode>=0.0){
                if(ielem == 0)  {
                //if(ielem == 7)  {

                    //myfile6<<ielem<<std::endl;
                    std::vector<double> eigenval_real(n_dof);
                    std::vector<double> eigenval_imag(n_dof);
                    for(unsigned int idof=0; idof<n_dof; idof++){
                        eigenval_real[idof] = eigen_solver.eigenvalues()[idof].real()/elem_wave_speed[ielem] * delx/ n_dof;
                        eigenval_imag[idof] = -eigen_solver.eigenvalues()[idof].imag()/elem_wave_speed[ielem] * delx / n_dof;
                    }
                    if(start_flag == false){
                        start_flag =true;
                        int index_phys = -1;
                        double min_dist= 1e6;
                        for(unsigned int idof=0; idof<n_dof; idof++){
                            double eig_dist = abs(eigenval_imag[idof] - fourier_mode/n_dof);
                            if(eig_dist<=1e-1 && eig_dist < min_dist && eigenval_imag[idof] >= 0.0){
                                index_phys = idof;
                                min_dist = eig_dist;
                            }
                        }
                        slope_prev = (eigenval_imag[index_phys])/(fourier_mode / n_dof);
                        slope_prev_diss = (eigenval_real[index_phys])/(fourier_mode / n_dof);
                        eig_prev = eigenval_imag[index_phys];
                        eig_prev_diss = eigenval_real[index_phys];
                        myfile6<<fourier_mode / n_dof<<std::endl;
                        myfile6<< std::fixed << std::setprecision(16) << eigenval_imag[index_phys] << " "<< std::fixed<<std::endl;
                        myfile6<< std::fixed << std::setprecision(16) << eigenval_real[index_phys] << " "<< std::fixed<<std::endl;
                    }
//                    else if(jump_flag == true){
//                        int index_phys = -1;
//                        double min_dist= 1e6;
//                        for(unsigned int idof=0; idof<n_dof; idof++){
//                            double eig_slope =(eigenval_imag[idof] - eig_prev) / (fourier_mode/n_dof - (fourier_mode-0.1)/n_dof) ;
//                            const double eig_dist = abs(eig_slope - slope_prev);
//                           // if(eig_dist < min_dist && eigenval_imag[idof] >= 0.0){
//                            if(eig_dist < min_dist){
//                                index_phys = idof;
//                                min_dist = eig_dist;
//                            }
//                        }
//                        slope_prev =(eigenval_imag[index_phys] - eig_prev) / (fourier_mode/n_dof - (fourier_mode-0.1)/n_dof) ;
//                        eig_prev = eigenval_imag[index_phys];
//                        if(slope_prev > 0){
//                            jump_flag = false;
//                        }
//                    }
                    else{
                        int index_phys = -1;
                        double min_dist= 1e6;
//                        if((slope_prev)<0.2 && fourier_mode/n_dof > 1.6){
//                            min_dist = -1e6;
//                            for(unsigned int idof=0; idof<n_dof; idof++){
//                                double eig_slope =(eigenval_imag[idof] - eig_prev) / (fourier_mode/n_dof - (fourier_mode-0.1)/n_dof) ;
//                                if(eig_slope > min_dist && eigenval_imag[idof] >= 0.0){//take the max slope
//                                    index_phys = idof;
//                                    min_dist = eig_slope;
//                                }
//                            }
//                         //   jump_flag=true;
//                            slope_prev =(eigenval_imag[index_phys] - eig_prev) / (fourier_mode/n_dof - (fourier_mode-0.1)/n_dof) ;
//                            eig_prev = eigenval_imag[index_phys];
//                        }
//                        else
                    {
                            for(unsigned int idof=0; idof<n_dof; idof++){
                                const double base = fourier_mode/n_dof - (fourier_mode-0.1)/n_dof;
                                const double eig_slope =(eigenval_imag[idof] - eig_prev) / (base) ;
                                const double eig_slope_diss =(eigenval_real[idof] - eig_prev_diss) / (base);
                                const double eig_dist = abs(eig_slope - slope_prev);
                                const double eig_dist_diss = abs(eig_slope_diss - slope_prev_diss);
                               // if(eig_dist < min_dist && eigenval_imag[idof] >= 0.0){
                                if((eig_dist + eig_dist_diss) < min_dist){
                                    index_phys = idof;
                                    min_dist = eig_dist + eig_dist_diss;
                                }
                            }
                            if(index_phys==-1)  std::abort();

                            slope_prev = (eigenval_imag[index_phys] - eig_prev) / (fourier_mode/n_dof - (fourier_mode-0.1)/n_dof);
                            eig_prev = eigenval_imag[index_phys];
                            slope_prev_diss =(eigenval_real[index_phys] - eig_prev_diss) / (fourier_mode/n_dof - (fourier_mode-0.1)/n_dof) ;
                            eig_prev_diss = eigenval_real[index_phys];
                        }
                        myfile6<<fourier_mode / n_dof<<std::endl;
                        myfile6<< std::fixed << std::setprecision(16) << eigenval_imag[index_phys] << " "<< std::fixed<<std::endl;
                        myfile6<< std::fixed << std::setprecision(16) << eigenval_real[index_phys] << " "<< std::fixed<<std::endl;


                    }




#if 0
                    int index_phys = -1;
                    double min_dist= 1e6;
                    for(unsigned int idof=0; idof<n_dof; idof++){
                        double eig_dist = abs(eigenval_imag[idof] - fourier_mode/n_dof);
                        if(eig_dist<=1e-1 && eig_dist < min_dist && eigenval_imag[idof] >= 0.0){
                            index_phys = idof;
                            min_dist = eig_dist;
                        }
                    }
                    if(index_phys != -1){
                        myfile6<< std::fixed << std::setprecision(16) << eigenval_imag[index_phys] << " "<< std::fixed<<std::endl;
                        myfile6<< std::fixed << std::setprecision(16) << eigenval_real[index_phys] << " "<< std::fixed<<std::endl;
                    }
                    else{
                        //get the eigenvectors, solve for coefficients, then find the mode that minimizes L2-error
                        Eigen::MatrixXcd eigenvectors(n_dof, n_dof);
                        Eigen::MatrixXcd eigenvectors_inv(n_dof, n_dof);
                        eigenvectors = eigen_solver.eigenvectors();
                        eigenvectors_inv = eigenvectors.inverse();
                        Eigen::VectorXcd eigenvector_coeff(n_dof);
                        for(unsigned int i=0; i<n_dof; i++){
                            eigenvector_coeff[i] = 0.0;
                            for(unsigned int j=0; j<n_dof; j++){
                                eigenvector_coeff[i] += eigenvectors_inv(i,j) * dg->solution[n_dof * ielem + j];
                            }
                        }

                        Eigen::MatrixXcd eigenvect_modes(n_dof, n_dof);
                        std::cout<<"egeiven vect modes sol "<<std::endl;
                   //     const std::vector<double> &vol_quad_weights = dg->volume_quadrature_collection[poly_degree].get_weights();
                   //     std::vector<std::complex<double>> avg_eigen_vect(n_dof);
                        for(unsigned int i=0; i<n_dof; i++){
                            for(unsigned int j=0; j<n_dof; j++){
                                eigenvect_modes(i,j) = eigenvector_coeff[j] * eigenvectors(i,j);
                   //             avg_eigen_vect[i] += eigenvector_coeff[i] * eigenvectors(j,i) * vol_quad_weights[j];
                            }
                        }
                        std::cout<<eigenvect_modes<<" "<<std::endl;
                        Eigen::MatrixXcd leg_eigenvectors(n_dof, n_dof);
                        for(unsigned int idof=0; idof<n_dof; idof++){
                            for(unsigned int jdof=0; jdof<n_dof; jdof++){
                                leg_eigenvectors(idof,jdof) = 0.0;
                                for(unsigned int kdof=0; kdof<n_dof; kdof++){
                                    leg_eigenvectors(idof,jdof) += eigenvect_modes(kdof, jdof) 
                                                                 * legendre_mat_inv(idof,kdof);
                                }
                            }
                        }
                        std::cout<<"legendre vect modes sol "<<std::endl;
                        std::cout<<leg_eigenvectors<<" "<<std::endl;
                        //look at most energized last mode
                        double max_val = -1.0;
                        for(unsigned int idof=0; idof<n_dof; idof++){
                            if(((leg_eigenvectors(idof,n_dof-1)).real()) > max_val && eigenval_imag[idof]>0){
                                max_val = (leg_eigenvectors(idof,n_dof-1).real());
                                index_phys = idof;
                            }
                        }
                        myfile6<< std::fixed << std::setprecision(16) << eigenval_imag[index_phys] << " "<< std::fixed<<std::endl;
                        myfile6<< std::fixed << std::setprecision(16) << eigenval_real[index_phys] << " "<< std::fixed<<std::endl;
#endif

#if 0
                        double l2_norm_min = 1e8;
                        int index_phys = -1;
                        for(unsigned int idof=0; idof<n_dof; idof++){
                            double l2_norm = 0.0;
#if 0
                           // for(unsigned int jdof=0; jdof<n_dof; jdof++){
                               unsigned int jdof = 0;
                                l2_norm += (eigenvect_modes(jdof,idof).real() - dg->solution[n_dof * ielem + jdof])
                                         * (eigenvect_modes(jdof,idof).real() - dg->solution[n_dof * ielem + jdof]);
                               jdof = n_dof-1;
                                l2_norm += (eigenvect_modes(jdof,idof).real() - dg->solution[n_dof * ielem + jdof])
                                         * (eigenvect_modes(jdof,idof).real() - dg->solution[n_dof * ielem + jdof]);
                           // }
#endif
                            l2_norm = (avg_eigen_vect[idof].real() - elem_wave_speed[ielem])
                                    * (avg_eigen_vect[idof].real() - elem_wave_speed[ielem]);
                            l2_norm = sqrt(l2_norm);
                            if(l2_norm < l2_norm_min && eigenval_imag[idof]>0){
                                l2_norm_min = l2_norm;
                                index_phys = idof;
                            }
                        }
                        myfile6<< std::fixed << std::setprecision(16) << eigenval_imag[index_phys] << " "<< std::fixed<<std::endl;
                        myfile6<< std::fixed << std::setprecision(16) << eigenval_real[index_phys] << " "<< std::fixed<<std::endl;

#endif

                     
#if 0                
                        //project onto Legendre poly
                        Eigen::MatrixXcd eigenvectors(n_dof, n_dof);
                        eigenvectors = eigen_solver.eigenvectors();
                        Eigen::MatrixXcd leg_eigenvectors(n_dof, n_dof);
                        for(unsigned int idof=0; idof<n_dof; idof++){
                            for(unsigned int jdof=0; jdof<n_dof; jdof++){
                                leg_eigenvectors(idof,jdof) = 0.0;
                                for(unsigned int kdof=0; kdof<n_dof; kdof++){
                                    leg_eigenvectors(idof,jdof) += eigenvectors(kdof, jdof) 
                                                                 * legendre_mat_inv(idof,kdof);
                                }
                            }
                        }
                        //look at most energized last mode
                        double max_val = -1.0;
                        for(unsigned int idof=0; idof<n_dof; idof++){
                            if(((leg_eigenvectors(n_dof-1,idof)).real()) > max_val){
                                max_val = (leg_eigenvectors(n_dof-1,idof).real());
                                index_phys = idof;
                            }
                        }
                        myfile6<< std::fixed << std::setprecision(16) << eigenval_imag[index_phys] << " "<< std::fixed<<std::endl;
                        myfile6<< std::fixed << std::setprecision(16) << eigenval_real[index_phys] << " "<< std::fixed<<std::endl;
#endif               
                //    }
                }

            }

        }


    }

//    myfile.close();
//    myfile2.close();
//    myfile3.close();
//    myfile4.close();
    myfile5.close();
    myfile6.close();
    return 0; //if got to here means passed the test, otherwise would've failed earlier
}

template class VonNeumannDispersionDissipation<PHILIP_DIM,1>;
#if PHILIP_DIM>1
template class VonNeumannDispersionDissipation<PHILIP_DIM,PHILIP_DIM>;
#endif
template class VonNeumannDispersionDissipation<PHILIP_DIM,PHILIP_DIM+2>;

} // Tests namespace
} // PHiLiP namespace


