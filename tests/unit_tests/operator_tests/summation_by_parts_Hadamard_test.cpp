#include <iomanip>
#include <cmath>
#include <limits>
#include <type_traits>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <bits/stdc++.h>

//#include <ctime>
#include <time.h>

#include <deal.II/distributed/solution_transfer.h>

#include "testing/tests.h"

#include<fstream>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/meshworker/dof_info.h>

// Finally, we take our exact solution from the library as well as volume_quadrature
// and additional tools.
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_fe_field.h> 

#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "operators/operators.h"

const double TOLERANCE = 1E-6;
using namespace std;


int main (int argc, char * argv[])
{

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    using real = double;
    using namespace PHiLiP;
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int dim = PHILIP_DIM;
    const int nstate = 1;
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);

    PHiLiP::Parameters::AllParameters all_parameters_new;
    all_parameters_new.parse_parameters (parameter_handler);
    all_parameters_new.nstate = nstate;
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    bool different = false;
    const unsigned int poly_max = 4;
    const unsigned int poly_min = 2;
    for(unsigned int poly_degree=poly_min; poly_degree<poly_max; poly_degree++){

        PHiLiP::OPERATOR::basis_functions<dim,2*dim> basis(nstate,poly_degree, 1);
        dealii::QGauss<1> quad1D (poly_degree+1);
        dealii::QGauss<0> quad1D_surf (poly_degree+1);
        const dealii::FE_DGQArbitraryNodes<1> fe_dg(quad1D);
        const dealii::FESystem<1,1> fe_system(fe_dg, 1);

        basis.build_1D_volume_operator(fe_system,quad1D);
        basis.build_1D_gradient_operator(fe_system,quad1D);
        basis.build_1D_surface_operator(fe_system,quad1D_surf);

        PHiLiP::OPERATOR::local_basis_stiffness<dim,2*dim> stiffness(nstate,poly_degree,1);
        stiffness.build_1D_volume_operator(fe_system,quad1D);

        PHiLiP::OPERATOR::local_mass<dim,2*dim> mass(nstate,poly_degree,1);
        mass.build_1D_volume_operator(fe_system,quad1D);

        PHiLiP::OPERATOR::surface_integral_SBP<dim,2*dim> surf_SBP(nstate,poly_degree,1);
        surf_SBP.build_1D_surface_operator(fe_system,quad1D_surf);

        PHiLiP::OPERATOR::vol_integral_basis<dim,2*dim> basis_int_vol(nstate,poly_degree,1);
        basis_int_vol.build_1D_volume_operator(fe_system,quad1D);

        const unsigned int n_dofs = nstate * pow(poly_degree+1,dim);

        for(unsigned int ielement=0; ielement<1; ielement++){//do several loops as if there were elements
            std::vector<real> sol_hat(n_dofs);
            for(unsigned int idof=0; idof<n_dofs; idof++){
                sol_hat[idof] = sqrt( 1e-8 + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(30-1e-8))) );
            }
            
            dealii::FullMatrix<real> sol_hat_mat(n_dofs);
            for(unsigned int idof=0; idof<n_dofs; idof++){
                for(unsigned int idof2=0; idof2<n_dofs; idof2++){
                    sol_hat_mat[idof][idof2] = sol_hat[idof] * sol_hat[idof2];
                }
            }
            dealii::Tensor<1,dim,dealii::FullMatrix<real>> dim_sol_hat_mat;
            for(int idim=0; idim<dim; idim++){
                dim_sol_hat_mat[idim].reinit(n_dofs, n_dofs);
                for(unsigned int idof=0; idof<n_dofs; idof++){
                    for(unsigned int idof2=0; idof2<n_dofs; idof2++){
                        dim_sol_hat_mat[idim][idof][idof2] = sol_hat_mat[idof][idof2];
                    }
                }
            }

            std::vector<real> div_vol_2pt_flux(n_dofs);

            basis.divergence_two_pt_flux_Hadamard_product(dim_sol_hat_mat, div_vol_2pt_flux, basis.oneD_grad_operator);

            std::vector<real> integrated_div_vol_2pt(n_dofs);
            std::vector<real> ones(n_dofs,1.0);
            basis_int_vol.inner_product_1D(div_vol_2pt_flux, ones, integrated_div_vol_2pt,
                                           basis_int_vol.oneD_vol_operator);


            dealii::Vector<real> stiff_trans_2pt(n_dofs);
            stiff_trans_2pt *= 0.0;
            for(int idim=0; idim<dim; idim++){
                dealii::FullMatrix<real> stiffness_transpose_dim(n_dofs);
                dealii::FullMatrix<real> stiffness_dim(n_dofs);
                if(idim==0){
                    stiffness_dim = stiffness.tensor_product(stiffness.oneD_vol_operator,
                                                             mass.oneD_vol_operator,
                                                             mass.oneD_vol_operator);
                    stiffness_transpose_dim.Tadd(1.0, stiffness_dim);
               //     stiffness_transpose_dim.add(1.0, stiffness_dim);
                }
                if(idim==1){
                    stiffness_dim = stiffness.tensor_product(mass.oneD_vol_operator,
                                                             stiffness.oneD_vol_operator,
                                                             mass.oneD_vol_operator);
                    stiffness_transpose_dim.Tadd(1.0, stiffness_dim);
                }
                if(idim==2){
                    stiffness_dim = stiffness.tensor_product(mass.oneD_vol_operator,
                                                             mass.oneD_vol_operator,
                                                             stiffness.oneD_vol_operator);
                    stiffness_transpose_dim.Tadd(1.0, stiffness_dim);
                }
                dealii::FullMatrix<real> hadamard_stiff_trans(n_dofs);
                for(unsigned int i=0; i<n_dofs; i++){
                    for(unsigned int j=0; j<n_dofs; j++){
                        hadamard_stiff_trans[i][j] = 2.0 * stiffness_transpose_dim[i][j] * sol_hat_mat[i][j]; 
                    }
                }
                dealii::Vector<real> ones_Vect(n_dofs);
                for(unsigned int i=0;i<n_dofs;i++){
                    ones_Vect[i] = 1.0;
                }
                hadamard_stiff_trans.vmult(stiff_trans_2pt, ones_Vect, true);
            }

            dealii::Vector<real> surf_2pt(n_dofs);
            surf_2pt *= 0.0;
            for(int idim=0; idim<dim; idim++){
                dealii::FullMatrix<real> surf_dim(n_dofs);
                if(idim==0){
                    surf_dim = surf_SBP.tensor_product(surf_SBP.oneD_surf_operator[1],
                                                       mass.oneD_vol_operator,
                                                       mass.oneD_vol_operator);
                    dealii::FullMatrix<real> temp(n_dofs);
                    temp = surf_SBP.tensor_product(surf_SBP.oneD_surf_operator[0],
                                                       mass.oneD_vol_operator,
                                                       mass.oneD_vol_operator);
                    surf_dim.add(-1.0, temp);
                }
                if(idim==1){
                    surf_dim = surf_SBP.tensor_product(mass.oneD_vol_operator,
                                                       surf_SBP.oneD_surf_operator[1],
                                                       mass.oneD_vol_operator);
                    dealii::FullMatrix<real> temp(n_dofs);
                    temp = surf_SBP.tensor_product(mass.oneD_vol_operator,
                                                       surf_SBP.oneD_surf_operator[0],
                                                       mass.oneD_vol_operator);
                    surf_dim.add(-1.0, temp);
                }
                if(idim==2){
                    surf_dim = surf_SBP.tensor_product(mass.oneD_vol_operator,
                                                       mass.oneD_vol_operator,
                                                       surf_SBP.oneD_surf_operator[1]);
                    dealii::FullMatrix<real> temp(n_dofs);
                    temp = surf_SBP.tensor_product(mass.oneD_vol_operator,
                                                       mass.oneD_vol_operator,
                                                       surf_SBP.oneD_surf_operator[0]);
                    surf_dim.add(-1.0, temp);
                }

                dealii::FullMatrix<real> hadamard_surf(n_dofs);
                for(unsigned int i=0; i<n_dofs; i++){
                    for(unsigned int j=0; j<n_dofs; j++){
                        hadamard_surf[i][j] = 2.0 * surf_dim[i][j] * sol_hat_mat[i][j]; 
                    }
                }
                dealii::Vector<real> ones_Vect(n_dofs);
                for(unsigned int i=0;i<n_dofs;i++){
                    ones_Vect[i] = 1.0;
                }
                hadamard_surf.vmult(surf_2pt, ones_Vect, true);

            }

            for(unsigned int i=0; i<n_dofs; i++){
                if(std::abs(surf_2pt[i] - (stiff_trans_2pt[i] + integrated_div_vol_2pt[i])) >1e-13)
                    different = true;
            }

        }//end of element loop
    }//end of poly_degree loop

    if(different==true){
        pcout<<"Hadamard product SBP not satisfied."<<std::endl;
        return 1;
    }
    else{
        return 0;
    }

}//end of main

