#include <fstream>
#include "dg/dg_factory.hpp"
#include "euler_split_inviscid_taylor_green_vortex.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "physics/initial_conditions/initial_condition_function.h"
#include "mesh/grids/nonsymmetric_curved_periodic_grid.hpp"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
EulerTaylorGreen<dim, nstate>::EulerTaylorGreen(const Parameters::AllParameters *const parameters_input)
    : TestsBase::TestsBase(parameters_input)
{}
template<int dim, int nstate>
std::array<double,2> EulerTaylorGreen<dim, nstate>::compute_change_in_entropy(const std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
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

    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_double  = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters));

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
        std::array<std::vector<double>,nstate> entropy_var_at_q;
        std::array<std::vector<double>,nstate> energy_var_at_q;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<double,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }
            std::array<double,nstate> entropy_var_state = euler_double->compute_entropy_variables(soln_state);
            std::array<double,nstate> kin_energy_state = euler_double->compute_kinetic_energy_variables(soln_state);
            for(int istate=0; istate<nstate; istate++){
                if(iquad==0){
                    entropy_var_at_q[istate].resize(n_quad_pts);
                    energy_var_at_q[istate].resize(n_quad_pts);
                }
                energy_var_at_q[istate][iquad] = kin_energy_state[istate];
                entropy_var_at_q[istate][iquad] = entropy_var_state[istate];
            }
        }
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
            }
        }
    }

    dg->assemble_residual();
    std::array<double,2> change_entropy_and_energy;
    change_entropy_and_energy[0] = entropy_var_hat_global * dg->right_hand_side;
    change_entropy_and_energy[1] = energy_var_hat_global * dg->right_hand_side;
    return change_entropy_and_energy;
}
template<int dim, int nstate>
double EulerTaylorGreen<dim, nstate>::compute_pressure_work(const std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
{
    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    const unsigned int grid_degree = dg->high_order_grid->fe_system.tensor_degree();

    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, dg->max_grid_degree);
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);
    soln_basis.build_1D_gradient_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);
    soln_basis.build_1D_surface_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_face_quadrature);

    OPERATOR::basis_functions<dim,2*dim> flux_basis(1, poly_degree, dg->max_grid_degree);
    flux_basis.build_1D_volume_operator(dg->oneD_fe_collection_flux[poly_degree], dg->oneD_quadrature_collection[poly_degree]);
    flux_basis.build_1D_gradient_operator(dg->oneD_fe_collection_flux[poly_degree], dg->oneD_quadrature_collection[poly_degree]);
    flux_basis.build_1D_surface_operator(dg->oneD_fe_collection_flux[poly_degree], dg->oneD_face_quadrature);

    OPERATOR::mapping_shape_functions<dim,2*dim> mapping_basis(1, poly_degree, grid_degree);
    mapping_basis.build_1D_shape_functions_at_grid_nodes(dg->high_order_grid->oneD_fe_system, dg->high_order_grid->oneD_grid_nodes);
    mapping_basis.build_1D_shape_functions_at_flux_nodes(dg->high_order_grid->oneD_fe_system, dg->oneD_quadrature_collection[poly_degree], dg->oneD_face_quadrature);
    const std::vector<double> &quad_weights_vol = dg->volume_quadrature_collection[poly_degree].get_weights();
    const std::vector<double> &quad_weights_surf = dg->face_quadrature_collection[poly_degree].get_weights();

    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_double  = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters));

    double pressure_work = 0.0;
    auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
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
        std::array<std::vector<double>,dim> vel_at_q;
        dealii::Tensor<1,dim,std::vector<double>> vel_grad_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            // Interpolate soln coeff to volume cubature nodes.
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }
        //get du/dx dv/dy, dw/dz
        for(int idim=0; idim<dim; idim++){
            dealii::Tensor<1,dim,std::vector<double>> ref_gradient_basis_fns_times_vel;
            for(int jdim=0; jdim<dim; jdim++){
                ref_gradient_basis_fns_times_vel[jdim].resize(n_quad_pts);
            }
            vel_at_q[idim].resize(n_quad_pts);
            vel_grad_at_q[idim].resize(n_quad_pts);
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                vel_at_q[idim][iquad] = soln_at_q[idim+1][iquad] / soln_at_q[0][iquad];
            }
            // Apply gradient of reference basis functions on the solution at volume cubature nodes.}
            flux_basis.gradient_matrix_vector_mult_1D(vel_at_q[idim], ref_gradient_basis_fns_times_vel,
                                                      flux_basis.oneD_vol_operator,
                                                      flux_basis.oneD_grad_operator);
            // Transform the reference gradient into a physical gradient operator.
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                for(int jdim=0; jdim<dim; jdim++){
                    //transform into the physical gradient
                    vel_grad_at_q[idim][iquad] += metric_oper.metric_cofactor_vol[idim][jdim][iquad]
                                                * ref_gradient_basis_fns_times_vel[jdim][iquad]
                                                / metric_oper.det_Jac_vol[iquad];
                }
            }
        }
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<double,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }
            const double pressure = euler_double->compute_pressure(soln_state);
            for(int idim=0; idim<dim; idim++){
                pressure_work += vel_grad_at_q[idim][iquad] * pressure * quad_weights_vol[iquad] * metric_oper.det_Jac_vol[iquad];
            }
        }
        const unsigned int n_quad_face_pts = dg->face_quadrature_collection[poly_degree].size();
        const unsigned int n_face_quad_pts = dg->face_quadrature_collection[poly_degree].size();
        for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {
            metric_oper.build_facet_metric_operators(
                iface,
                n_quad_face_pts, n_metric_dofs/dim,
                mapping_support_points,
                mapping_basis,
                dg->all_parameters->use_invariant_curl_form);
            const dealii::Tensor<1,dim,double> unit_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[iface];
            std::vector<dealii::Tensor<1,dim,double>> normals_int(n_quad_face_pts);
            for(unsigned int iquad=0; iquad<n_quad_face_pts; iquad++){
                for(unsigned int idim=0; idim<dim; idim++){
                    normals_int[iquad][idim] =  0.0;
                    for(int idim2=0; idim2<dim; idim2++){
                        normals_int[iquad][idim] += unit_normal_int[idim2] * metric_oper.metric_cofactor_surf[idim][idim2][iquad];//\hat{n}^r * C_m^T 
                    }
                }
            }
            const auto neighbor_cell = cell->neighbor_or_periodic_neighbor(iface);
            unsigned int neighbor_iface;
            auto current_face = cell->face(iface);
            if(current_face->at_boundary())
                neighbor_iface = cell->periodic_neighbor_of_periodic_neighbor(iface);
            else
                neighbor_iface = cell->neighbor_of_neighbor(iface);

            // Get information about neighbor cell
            const unsigned int n_dofs_neigh_cell = dg->fe_collection[neighbor_cell->active_fe_index()].n_dofs_per_cell();
            // Obtain the mapping from local dof indices to global dof indices for neighbor cell
            std::vector<dealii::types::global_dof_index> neighbor_dofs_indices;
            neighbor_dofs_indices.resize(n_dofs_neigh_cell);
            neighbor_cell->get_dof_indices (neighbor_dofs_indices);
             
            const int poly_degree_ext = neighbor_cell->active_fe_index();
            std::array<std::vector<double>,nstate> soln_coeff_ext;
            for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                const unsigned int istate = dg->fe_collection[poly_degree_ext].system_to_component_index(idof).first;
                const unsigned int ishape = dg->fe_collection[poly_degree_ext].system_to_component_index(idof).second;
                if(ishape == 0)
                    soln_coeff_ext[istate].resize(n_shape_fns);
                soln_coeff_ext[istate][ishape] = dg->solution(neighbor_dofs_indices[idof]);
            }
            std::array<std::vector<double>,nstate> soln_at_q_ext;
            for(int istate=0; istate<nstate; istate++){
                soln_at_q_ext[istate].resize(n_quad_pts);
                // Interpolate soln coeff to volume cubature nodes.
                soln_basis.matrix_vector_mult_1D(soln_coeff_ext[istate], soln_at_q_ext[istate],
                                                 soln_basis.oneD_vol_operator);
            }

            //get volume entropy var and interp to face
            std::array<std::vector<double>,nstate> entropy_var_vol_int;
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                std::array<double,nstate> soln_state;
                for(int istate=0; istate<nstate; istate++){
                    soln_state[istate] = soln_at_q[istate][iquad];
                }
                std::array<double,nstate> entropy_var;
                entropy_var = euler_double->compute_entropy_variables(soln_state);
                for(int istate=0; istate<nstate; istate++){
                    if(iquad==0){
                        entropy_var_vol_int[istate].resize(n_quad_pts);
                    }
                    entropy_var_vol_int[istate][iquad] = entropy_var[istate];
                }
            }
            std::array<std::vector<double>,nstate> entropy_var_vol_ext;
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                std::array<double,nstate> soln_state;
                for(int istate=0; istate<nstate; istate++){
                    soln_state[istate] = soln_at_q_ext[istate][iquad];
                }
                std::array<double,nstate> entropy_var;
                entropy_var = euler_double->compute_entropy_variables(soln_state);
                for(int istate=0; istate<nstate; istate++){
                    if(iquad==0){
                        entropy_var_vol_ext[istate].resize(n_quad_pts);
                    }
                    entropy_var_vol_ext[istate][iquad] = entropy_var[istate];
                }
            }
            //Then interpolate the entropy variables at volume cubature nodes to the facet.
            std::array<std::vector<double>,nstate> entropy_var_vol_int_interp_to_surf;
            std::array<std::vector<double>,nstate> entropy_var_vol_ext_interp_to_surf;
            for(int istate=0; istate<nstate; ++istate){
                // allocate
                entropy_var_vol_int_interp_to_surf[istate].resize(n_face_quad_pts);
                entropy_var_vol_ext_interp_to_surf[istate].resize(n_face_quad_pts);
                // solve entropy variables at facet cubature nodes
                flux_basis.matrix_vector_mult_surface_1D(iface,
                                                         entropy_var_vol_int[istate], 
                                                         entropy_var_vol_int_interp_to_surf[istate],
                                                         flux_basis.oneD_surf_operator,
                                                         flux_basis.oneD_vol_operator);
                flux_basis.matrix_vector_mult_surface_1D(neighbor_iface,
                                                         entropy_var_vol_ext[istate], 
                                                         entropy_var_vol_ext_interp_to_surf[istate],
                                                         flux_basis.oneD_surf_operator,
                                                         flux_basis.oneD_vol_operator);
            }


            //end of get volume entropy var and interp to face
//            std::array<std::vector<double>,nstate> soln_at_surf_q_int;
//            std::array<std::vector<double>,nstate> soln_at_surf_q_ext;
//            for(int istate=0; istate<nstate; ++istate){
//                // allocate
//                soln_at_surf_q_int[istate].resize(n_face_quad_pts);
//                soln_at_surf_q_ext[istate].resize(n_face_quad_pts);
//                // solve soln at facet cubature nodes
//                soln_basis.matrix_vector_mult_surface_1D(iface,
//                                                         soln_coeff[istate], soln_at_surf_q_int[istate],
//                                                         soln_basis.oneD_surf_operator,
//                                                         soln_basis.oneD_vol_operator);
//                soln_basis.matrix_vector_mult_surface_1D(neighbor_iface,
//                                                         soln_coeff_ext[istate], soln_at_surf_q_ext[istate],
//                                                         soln_basis.oneD_surf_operator,
//                                                         soln_basis.oneD_vol_operator);
//            }

            std::array<std::vector<double>,dim> vel_at_surf_q_int;
            for(int idim=0; idim<dim; ++idim){
                // allocate
                vel_at_surf_q_int[idim].resize(n_face_quad_pts);
                // solve soln at facet cubature nodes
                flux_basis.matrix_vector_mult_surface_1D(iface,
                                                         vel_at_q[idim], vel_at_surf_q_int[idim],
                                                         flux_basis.oneD_surf_operator,
                                                         flux_basis.oneD_vol_operator);
            }
            for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
//                std::array<double,nstate> soln_state_int;
//                std::array<double,nstate> soln_state_ext;
//                for(int istate=0; istate<nstate; istate++){
//                    soln_state_int[istate] = soln_at_surf_q_int[istate][iquad];
//                    soln_state_ext[istate] = soln_at_surf_q_ext[istate][iquad];
//                }
                std::array<double,nstate> entropy_var_face_int;
                std::array<double,nstate> entropy_var_face_ext;
                for(int istate=0; istate<nstate; istate++){
                    entropy_var_face_int[istate] = entropy_var_vol_int_interp_to_surf[istate][iquad];
                    entropy_var_face_ext[istate] = entropy_var_vol_ext_interp_to_surf[istate][iquad];
                }

                std::array<double,nstate> soln_state_int;
                soln_state_int = euler_double->compute_conservative_variables_from_entropy_variables (entropy_var_face_int);
                std::array<double,nstate> soln_state_ext;
                soln_state_ext = euler_double->compute_conservative_variables_from_entropy_variables (entropy_var_face_ext);
                
                const double pressure_int = euler_double->compute_pressure(soln_state_int);
                const double pressure_ext = euler_double->compute_pressure(soln_state_ext);
                for(int idim=0; idim<dim; idim++){
                  //  double vel_int = soln_at_surf_q_int[idim+1][iquad] / soln_at_surf_q_int[0][iquad];
                   // double vel_int = soln_state_int[idim+1] / soln_state_int[0];
                    double vel_int = vel_at_surf_q_int[idim][iquad];
//                    double vel_ext = soln_at_surf_q_ext[idim+1][iquad] / soln_at_surf_q_ext[0][iquad];

                   // pressure_work -= quad_weights_surf[iquad] * 0.5*(pressure_int + pressure_ext) * normals_int[iquad][idim] * (vel_int -vel_ext); 
                   //only do interior since double count face
                    pressure_work -= quad_weights_surf[iquad] * 0.5*(pressure_int + pressure_ext) * normals_int[iquad][idim] * vel_int; 
                }
            }
        }
    }

    return pressure_work;
}

template<int dim, int nstate>
double EulerTaylorGreen<dim, nstate>::compute_entropy(const std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
{
    //returns the entropy evaluated in the broken Sobolev-norm rather than L2-norm
    dealii::LinearAlgebra::distributed::Vector<double> mass_matrix_times_solution(dg->right_hand_side);
    if(dg->all_parameters->use_inverse_mass_on_the_fly)
        dg->apply_global_mass_matrix(dg->solution,mass_matrix_times_solution);
    else
        dg->global_mass_matrix.vmult( mass_matrix_times_solution, dg->solution);

    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    //We have to project the vector of entropy variables because the mass matrix has an interpolation from solution nodes built into it.
    OPERATOR::vol_projection_operator<dim,2*dim> vol_projection(1, poly_degree, dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, dg->max_grid_degree);
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global(dg->right_hand_side);
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

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
        std::array<std::vector<double>,nstate> entropy_var_at_q;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<double,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }
            std::array<double,nstate> entropy_var_state = pde_physics_double->compute_entropy_variables(soln_state);
            for(int istate=0; istate<nstate; istate++){
                if(iquad==0)
                    entropy_var_at_q[istate].resize(n_quad_pts);
                entropy_var_at_q[istate][iquad] = entropy_var_state[istate];
            }
        }
        for(int istate=0; istate<nstate; istate++){
            //Projected vector of entropy variables.
            std::vector<double> entropy_var_hat(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(entropy_var_at_q[istate], entropy_var_hat,
                                                 vol_projection.oneD_vol_operator);

            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                const unsigned int idof = istate * n_shape_fns + ishape;
                entropy_var_hat_global[dofs_indices[idof]] = entropy_var_hat[ishape];
            }
        }
    }

    double entropy = entropy_var_hat_global * mass_matrix_times_solution;
    return entropy;
}

template<int dim, int nstate>
double EulerTaylorGreen<dim, nstate>::compute_kinetic_energy(const std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
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

    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_double  = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters));

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

     //       const double quadrature_kinetic_energy =  0.5*(soln_at_q[1]*soln_at_q[1]
     //                                               + soln_at_q[2]*soln_at_q[2]
     //                                               + soln_at_q[3]*soln_at_q[3])/density;
            const double pressure = euler_double->compute_pressure(soln_at_q);
            const double entropy = log(pressure) - 1.4 * log(density);
            const double quadrature_kinetic_energy = -density*entropy/0.4;

            total_kinetic_energy += quadrature_kinetic_energy * fe_values_extra.JxW(iquad);
        }
    }
    return total_kinetic_energy;
}

template<int dim, int nstate>
double EulerTaylorGreen<dim, nstate>::get_timestep(const std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree, const double delta_x) const
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

        const double max_eig_mpi = dealii::Utilities::MPI::max(max_eig, mpi_communicator);
        double cfl = 0.1 * delta_x/max_eig_mpi;
        if(cfl < cfl_min)
            cfl_min = cfl;

    }
    return cfl_min;
}

template <int dim, int nstate>
int EulerTaylorGreen<dim, nstate>::run_test() const
{
    // using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    // std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (mpi_communicator);
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    using real = double;

    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;  
    double left = 0.0;
    double right = 2 * dealii::numbers::PI;
    const int n_refinements = 2;
    unsigned int poly_degree = 3;

    // set the warped grid
//    const unsigned int grid_degree = poly_degree;
//    PHiLiP::Grids::nonsymmetric_curved_grid<dim,Triangulation>(*grid, n_refinements);

    const unsigned int grid_degree = 1;
    const bool colorize = true;
    dealii::GridGenerator::hyper_cube(*grid, left, right, colorize);
    std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
    dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
    dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
    dealii::GridTools::collect_periodic_faces(*grid,4,5,2,matched_pairs);
    grid->add_periodicity(matched_pairs);
    grid->refine_global(n_refinements);

    // Create DG
    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
    dg->allocate_system ();

    std::cout << "Implement initial conditions" << std::endl;
    std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function = 
                InitialConditionFactory<dim,nstate,double>::create_InitialConditionFunction(&all_parameters_new);
    SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);

    const unsigned int n_global_active_cells2 = grid->n_global_active_cells();
    double delta_x = (right-left)/n_global_active_cells2/(poly_degree+1.0);
    pcout<<" delta x "<<delta_x<<std::endl;

    all_parameters_new.ode_solver_param.initial_time_step =  get_timestep(dg,poly_degree,delta_x);
     
    std::cout << "creating ODE solver" << std::endl;
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    std::cout << "ODE solver successfully created" << std::endl;
    double finalTime = 14.;
    finalTime = 0.4;
    // finalTime = 0.1;//to speed things up locally in tests, doesn't need full 14seconds to verify.
    double dt = all_parameters_new.ode_solver_param.initial_time_step;
    // double dt = all_parameters_new.ode_solver_param.initial_time_step / 10.0;
//    finalTime = 14.0;
//    finalTime = all_parameters_new.ode_solver_param.initial_time_step;

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
    double initial_entropy = compute_entropy(dg, poly_degree);
    pcout<<"Initial MK Entropy "<<initial_entropy<<std::endl;
    std::array<double,2> initial_change_entropy = compute_change_in_entropy(dg, poly_degree);
    pcout<<"Initial change in Entropy "<<initial_change_entropy[0]<<std::endl;
    pcout<<"Initial change in kinetic Energy "<<initial_change_entropy[1]<<std::endl;
    double initial_p_work = compute_pressure_work(dg, poly_degree);
    double initial_p_work_mpi = (dealii::Utilities::MPI::sum(initial_p_work, mpi_communicator));
    pcout<<"Initial pressure work "<<initial_p_work_mpi<<std::endl;
    pcout<<"Initial change energy pressure work "<<initial_change_entropy[1]-initial_p_work_mpi<<std::endl;

    std::cout << std::setprecision(16) << std::fixed;
    pcout << "Energy at one timestep is " << initial_energy/(8*pow(dealii::numbers::PI,3)) << std::endl;
    // std::ofstream myfile ("kinetic_energy_3D_TGV_cdg_curv_grid_4x4.gpl" , std::ios::trunc);
    std::ofstream myfile (all_parameters_new.energy_file + ".gpl"  , std::ios::trunc);

    for (int i = 0; i < std::ceil(finalTime/dt); ++ i) {
        ode_solver->advance_solution_time(dt);
        double current_energy = compute_kinetic_energy(dg,poly_degree);
        std::cout << std::setprecision(16) << std::fixed;
        pcout << "Energy at time " << i * dt << " is " << current_energy / initial_energy << std::endl;
        pcout << "Actual Energy Divided by volume at time " << i * dt << " is " << current_energy/(8*pow(dealii::numbers::PI,3)) << std::endl;
        if (current_energy - initial_energy >= 1.00)
        {
          pcout << " Energy was not monotonically decreasing" << std::endl;
          return 1;
        }
        double current_entropy = compute_entropy(dg, poly_degree);
        std::cout << std::setprecision(16) << std::fixed;
        pcout << "M plus K norm Entropy at time " << i * dt << " is " << current_entropy / initial_entropy<< std::endl;
//        myfile << i * dt << " " << std::fixed << std::setprecision(16) << current_entropy / initial_entropy<< std::endl;

        std::array<double,2> current_change_entropy = compute_change_in_entropy(dg, poly_degree);
        std::cout << std::setprecision(16) << std::fixed;
        pcout << "M plus K norm Change in Entropy at time " << i * dt << " is " << current_change_entropy[0]<< std::endl;
        pcout << "M plus K norm Change in Kinetic Energy at time " << i * dt << " is " << current_change_entropy[1]<< std::endl;
        if(abs(current_change_entropy[0]) > 1e-12 && (dg->all_parameters->two_point_num_flux_type == Parameters::AllParameters::TwoPointNumericalFlux::IR || dg->all_parameters->two_point_num_flux_type == Parameters::AllParameters::TwoPointNumericalFlux::CH || dg->all_parameters->two_point_num_flux_type == Parameters::AllParameters::TwoPointNumericalFlux::Ra)){
          pcout << " Entropy was not monotonically decreasing" << std::endl;
          return 1;
        }
        myfile << i * dt << " " << std::fixed << std::setprecision(16) << current_change_entropy[0]<< std::endl;
//        myfile << i * dt << " " << std::fixed << std::setprecision(16) << current_change_entropy[1]<< std::endl;
        double current_p_work = compute_pressure_work(dg, poly_degree);
        double current_p_work_mpi = (dealii::Utilities::MPI::sum(current_p_work, mpi_communicator));
        pcout<<"Current change energy pressure work "<<current_change_entropy[1]-current_p_work_mpi<<std::endl;
        myfile << i * dt << " " << std::fixed << std::setprecision(16) << current_change_entropy[1]-current_p_work_mpi<< std::endl;
        all_parameters_new.ode_solver_param.initial_time_step =  get_timestep(dg,poly_degree, delta_x);
    }

    myfile.close();




    return 0;
}

#if PHILIP_DIM==3
    template class EulerTaylorGreen <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace
