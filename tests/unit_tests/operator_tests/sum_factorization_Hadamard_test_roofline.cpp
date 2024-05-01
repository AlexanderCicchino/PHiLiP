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


void compute_Hadamard(const unsigned int n_quad_pts,
                      const unsigned int n_dofs_1D,
                      PHiLiP::OPERATOR::basis_functions<3,6,double> &basis,
                      dealii::FullMatrix<double> &sol_dim,
                      dealii::FullMatrix<double> &sol_hat_mat,
                      std::vector<double> &output)
{
    dealii::FullMatrix<double> basis_dim(n_quad_pts);//solution of A*u with sum-factorization
    basis_dim = basis.tensor_product(basis.oneD_grad_operator, basis.oneD_vol_operator, basis.oneD_vol_operator);
    for(unsigned int idof=0; idof< n_dofs_1D * n_dofs_1D * n_dofs_1D; idof++){
        for(unsigned int idof2=0; idof2< n_dofs_1D * n_dofs_1D * n_dofs_1D; idof2++){
        sol_dim[idof][idof2] = sol_hat_mat[idof][idof2]
                             * basis_dim[idof][idof2];
        }
    }
    for(unsigned int idof=0; idof< n_dofs_1D * n_dofs_1D * n_dofs_1D; idof++){
        for(unsigned int idof2=0; idof2< n_dofs_1D * n_dofs_1D * n_dofs_1D; idof2++){
            output[idof] += sol_dim[idof][idof2];
        }
    }
    basis_dim = basis.tensor_product(basis.oneD_vol_operator, basis.oneD_grad_operator, basis.oneD_vol_operator);
    for(unsigned int idof=0; idof< n_dofs_1D * n_dofs_1D * n_dofs_1D; idof++){
        for(unsigned int idof2=0; idof2< n_dofs_1D * n_dofs_1D * n_dofs_1D; idof2++){
        sol_dim[idof][idof2] = sol_hat_mat[idof][idof2]
                             * basis_dim[idof][idof2];
        }
    }
    for(unsigned int idof=0; idof< n_dofs_1D * n_dofs_1D * n_dofs_1D; idof++){
        for(unsigned int idof2=0; idof2< n_dofs_1D * n_dofs_1D * n_dofs_1D; idof2++){
            output[idof] += sol_dim[idof][idof2];
        }
    }
    basis_dim = basis.tensor_product(basis.oneD_vol_operator, basis.oneD_vol_operator, basis.oneD_grad_operator);
    for(unsigned int idof=0; idof< n_dofs_1D * n_dofs_1D * n_dofs_1D; idof++){
        for(unsigned int idof2=0; idof2< n_dofs_1D * n_dofs_1D * n_dofs_1D; idof2++){
        sol_dim[idof][idof2] = sol_hat_mat[idof][idof2]
                             * basis_dim[idof][idof2];
        }
    }
    for(unsigned int idof=0; idof< n_dofs_1D * n_dofs_1D * n_dofs_1D; idof++){
        for(unsigned int idof2=0; idof2< n_dofs_1D * n_dofs_1D * n_dofs_1D; idof2++){
            output[idof] += sol_dim[idof][idof2];
        }
    }

}


void compute_Hadamard_sum_fact(PHiLiP::OPERATOR::basis_functions<3,2*3,double> &basis,
                               dealii::FullMatrix<double> &sol_hat_mat,
                               dealii::FullMatrix<double> &sol_1D,
                               std::vector<double> &ones)
{

    basis.two_pt_flux_Hadamard_product(sol_hat_mat, sol_1D, basis.oneD_grad_operator, ones, 0);
    basis.two_pt_flux_Hadamard_product(sol_hat_mat, sol_1D, basis.oneD_grad_operator, ones, 1);
    basis.two_pt_flux_Hadamard_product(sol_hat_mat, sol_1D, basis.oneD_grad_operator, ones, 2);

}


void poly_degree_Hadamard(const unsigned int /*poly_degree*/,
                          const unsigned int /*nstate*/,
                          const unsigned int /*n_quad_pts_1D*/,
                          const unsigned int n_quad_pts,
                          const unsigned int n_dofs,
                          const unsigned int n_dofs_1D,
                          std::vector<double> &output,
                          const std::vector<double> &sol_hat,
                          PHiLiP::OPERATOR::basis_functions<3,2*3,double> &basis)
{
    
    
    dealii::FullMatrix<double> sol_hat_mat(n_dofs);
    for(unsigned int idof=0; idof<n_dofs; idof++){
        for(unsigned int idof2=0; idof2<n_dofs; idof2++){
            sol_hat_mat[idof][idof2] = sol_hat[idof] * sol_hat[idof2];
        }
    }
     
    dealii::FullMatrix<double> sol_dim(n_quad_pts);//solution of A*u normally
    dealii::FullMatrix<double> sol_1D(n_quad_pts);//solution of A*u with sum-factorization

    for(unsigned int i=0;i<10;i++){
        compute_Hadamard(n_quad_pts, n_dofs_1D, basis, sol_dim, sol_hat_mat,output);
    }

}

template<int nstate>
std::array<dealii::Tensor<1,3,double>,nstate> two_pt_flux_for_test(
    const std::array<double,nstate> &sol1, const std::array<double,nstate> sol2)
{
    std::array<dealii::Tensor<1,3,double>,nstate> out;
    for(int istate=0; istate<nstate;istate++){
        for(int idim=0; idim<3; idim++){
            out[istate][idim] = sol1[istate] * sol2[istate];
        }
    }
    return out;

}

void poly_degree_Hadamard_sum_fact(const unsigned int /*poly_degree*/,
                                   const unsigned int /*nstate*/,
                                   std::array<std::vector<double>,1> &output,
                                   const std::array<std::vector<double>,1> &sol_hat,
                                   const std::vector<double> &ones,
                                   const std::vector<double> &ones_dim,
                                   PHiLiP::OPERATOR::basis_functions<3,2*3,double> &basis)
{
    
    const unsigned int size = basis.oneD_grad_operator.n();
  //  std::vector<std::vector<double>> grad_oper(size);
  //  for(unsigned int i=0;i<size;i++){
  //      grad_oper[i].resize(size);
  //      for(unsigned int j=0;j<size;j++){
  //          grad_oper[i][j] = basis.oneD_grad_operator[i][j];
  //      }
  //  }

    for(unsigned int i=0;i<100;i++){
        basis.two_pt_flux_Hadamard_product_sparsity_on_the_fly<1>(ones_dim,
            sol_hat, &two_pt_flux_for_test<1>, output, basis.oneD_grad_operator, size, ones, 1.0);
           // sol_hat, &two_pt_flux_for_test<1>, output, grad_oper,size, ones, 1.0);
    }
                                        
    
//    dealii::FullMatrix<double> sol_hat_mat(n_dofs);
//    for(unsigned int idof=0; idof<n_dofs; idof++){
//        for(unsigned int idof2=0; idof2<n_dofs; idof2++){
//            sol_hat_mat[idof][idof2] = sol_hat[idof] * sol_hat[idof2];
//        }
//    }
//     
//    dealii::FullMatrix<double> sol_dim(n_quad_pts);//solution of A*u normally
//    dealii::FullMatrix<double> sol_1D(n_quad_pts);//solution of A*u with sum-factorization
//
//   // for(unsigned int i=0;i<100;i++){
//        compute_Hadamard_sum_fact(basis, sol_hat_mat, sol_1D, ones);
//   // }
//
//
}

int main (int argc, char * argv[])
{

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    using real = double;
    using namespace PHiLiP;
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int nstate = 1;
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler);

    PHiLiP::Parameters::AllParameters all_parameters_new;
    all_parameters_new.parse_parameters (parameter_handler);
    all_parameters_new.nstate = nstate;
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);


    const unsigned int poly_max = 15;
    const unsigned int poly_min = 14;

    for(unsigned int poly_degree=poly_min; poly_degree<poly_max; poly_degree++){

        const unsigned int size = (poly_degree+1)* (poly_degree+1)* (poly_degree+1);
        std::vector<double> output_had_basic(size);
        std::array<std::vector<double>,1> output_had_sumfact;
        output_had_sumfact[0].resize(size);
        PHiLiP::OPERATOR::basis_functions<3,2*3,double> basis(1,poly_degree, 1);
        dealii::QGauss<1> quad1D (poly_degree+1);
        const dealii::FE_DGQArbitraryNodes<1> fe_dg(quad1D);
        const dealii::FESystem<1,1> fe_system(fe_dg, 1);
        basis.build_1D_volume_operator(fe_system,quad1D);
        basis.build_1D_gradient_operator(fe_system,quad1D);
        const unsigned int n_dofs = nstate * pow(poly_degree+1,3);
        const unsigned int n_dofs_1D = nstate * (poly_degree+1);
        const unsigned int n_quad_pts_1D = quad1D.size();
        const unsigned int n_quad_pts = pow(n_quad_pts_1D, 3);
        std::vector<double> ones(n_quad_pts_1D, 1.0);
        std::vector<double> ones_dim(n_quad_pts, 1.0);
        std::vector<double> sol_hat(n_dofs);
        std::array<std::vector<double>,1> sol_hat_arr;
        sol_hat_arr[0].resize(n_dofs);
        for(unsigned int idof=0; idof<n_dofs; idof++){
            sol_hat[idof] = sqrt( 1e-8 + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(2.0-1e-8))) );
            sol_hat_arr[0][idof] = sol_hat[idof];
        }
        poly_degree_Hadamard(poly_degree, nstate,n_quad_pts_1D,n_quad_pts,n_dofs,n_dofs_1D,output_had_basic,sol_hat,basis);
       // poly_degree_Hadamard_sum_fact(poly_degree, nstate,output_had_sumfact,sol_hat_arr,ones,ones_dim,basis);
//        for(unsigned int i=; i<size;i++){
//            std::cout<<"output basic "<<output_had_basic[i]<<" vs "<<output_had_sumfact[0][i]<<std::endl;
//            double a = output_had_basic[i];
//            double b = output_had_sumfact[0][i];
//            if(abs(output_had_basic[i]-output_had_sumfact[0][i])>1e-13){
//            if(abs(a - b) > pow(10.0, 3) * std::numeric_limits<double>::epsilon() * std::max(abs(a), abs(b))){
//                std::cout<<"they aren't equal"<<std::endl;
//                std::cout<<" first "<<abs(a - b)<<" vs "<<pow(10.0, 3) * std::numeric_limits<double>::epsilon() * std::max(abs(a), abs(b))<<std::endl;
//                std::cout<<"output basic "<<output_had_basic[i]<<" vs "<<output_had_sumfact[0][i]<<std::endl;
//                return 1;
//            }
//        }

    }//end of poly_degree loop

    return 0;

}//end of main

