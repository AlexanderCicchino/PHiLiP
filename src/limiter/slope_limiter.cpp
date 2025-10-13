#include "positivity_preserving_limiter.h"
#include "tvb_limiter.h"
#include "slope_limiter.h"

namespace PHiLiP {
/**********************************
*
* Positivity Preserving Limiter Class
*
**********************************/
// Constructor
template <int dim, int nstate, typename real>
SlopeLimiter<dim, nstate, real>::SlopeLimiter(
    const Parameters::AllParameters* const parameters_input)
    : BoundPreservingLimiterState<dim,nstate,real>::BoundPreservingLimiterState(parameters_input)
{
    // Create pointer to Euler Physics to compute pressure if pde_type==euler
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    pde_type = parameters_input->pde_type;


    if (pde_type == PDE_enum::euler && nstate == dim + 2) {
        std::shared_ptr< ManufacturedSolutionFunction<dim, real> >  manufactured_solution_function
            = ManufacturedSolutionFactory<dim, real>::create_ManufacturedSolution(parameters_input, nstate);
        euler_physics = std::make_shared < Physics::Euler<dim, nstate, real> >(
            parameters_input,
            parameters_input->euler_param.ref_length,
            parameters_input->euler_param.gamma_gas,
            parameters_input->euler_param.mach_inf,
            parameters_input->euler_param.angle_of_attack,
            parameters_input->euler_param.side_slip_angle,
            manufactured_solution_function,
            parameters_input->two_point_num_flux_type);
    }
    else if (pde_type == PDE_enum::burgers_inviscid && dim == 1) {
        std::shared_ptr< ManufacturedSolutionFunction<1, real> >  manufactured_solution_function
            = ManufacturedSolutionFactory<1, real>::create_ManufacturedSolution(parameters_input, 1);
        burgers_physics = std::make_shared < Physics::Burgers<1, 1, real> >(
                parameters_input,
                parameters_input->burgers_param.diffusion_coefficient,
                true, false,
                parameters_input->manufactured_convergence_study_param.manufactured_solution_param.diffusion_tensor, 
                manufactured_solution_function,
                parameters_input->test_type);
    }
    else {
        std::cout << "Error: Slope Limiter can only be applied for pde_type==euler or burgers_inviscid" << std::endl;
        std::abort();
    }
    this->posdensity_limiter = std::make_shared < PositivityPreservingLimiter<dim, nstate, real> >(parameters_input);
}

template <int dim, int nstate, typename real>
void SlopeLimiter<dim, nstate, real>::set_cell_min_entropy(
    dealii::LinearAlgebra::distributed::Vector<double>&     solution,
    const dealii::DoFHandler<dim>&                          dof_handler,
    const dealii::hp::FECollection<dim>&                    fe_collection,
    const dealii::hp::QCollection<dim>&                     volume_quadrature_collection,
    const unsigned int                                      grid_degree,
    const unsigned int                                      max_degree,
    const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
    const dealii::hp::QCollection<1>                        oneD_quadrature_collection)
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;

    unsigned int n_cells_local=0;
    for (auto soln_cell : dof_handler.active_cell_iterators()) {
        if (!soln_cell->is_locally_owned()) continue;
        n_cells_local++;
    }
    local_min_entropy.reinit(n_cells_local);


    const unsigned int init_grid_degree = grid_degree;

    // Constructor for the operators
    OPERATOR::basis_functions<dim, 2 * dim, real> soln_basis(1, max_degree, init_grid_degree);
    OPERATOR::vol_projection_operator<dim, 2 * dim, real> soln_basis_projection_oper(1, max_degree, init_grid_degree);

    // Build the oneD operator to perform interpolation/projection
    soln_basis.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quadrature_collection[max_degree]);
    soln_basis_projection_oper.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_fe_collection_1state[max_degree],oneD_quadrature_collection[max_degree]);

    /*loop over cells and compute limiter for min enetorpy principle
    *   then apply the limiter
    */
    unsigned int cell_index = 0;
    for (auto soln_cell : dof_handler.active_cell_iterators()) {
        if (!soln_cell->is_locally_owned()) continue;

        std::vector<dealii::types::global_dof_index> current_dofs_indices;
        // Current reference element related to this physical cell
        const int i_fele = soln_cell->active_fe_index();
        const dealii::FESystem<dim, dim>& current_fe_ref = fe_collection[i_fele];
        const int poly_degree = current_fe_ref.tensor_degree();

        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

        // Obtain the mapping from local dof indices to global dof indices
        current_dofs_indices.resize(n_dofs_curr_cell);
        soln_cell->get_dof_indices(current_dofs_indices);

        // Extract the local solution dofs in the cell from the global solution dofs
        std::array<std::vector<real>, nstate> soln_coeff;

        const unsigned int n_shape_fns = n_dofs_curr_cell / nstate;
        for (unsigned int istate = 0; istate < nstate; ++istate) {
            soln_coeff[istate].resize(n_shape_fns);
        }

        for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
            const unsigned int istate = fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = fe_collection[poly_degree].system_to_component_index(idof).second;
            soln_coeff[istate][ishape] = solution[current_dofs_indices[idof]]; //
        }

        const unsigned int n_quad_pts = volume_quadrature_collection[poly_degree].size();
        std::array<std::vector<real>, nstate> soln_at_q;

        // Interpolate solution dofs to quadrature pts.
        for (int istate = 0; istate < nstate; istate++) {
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }

        // Obtain solution cell average
        real local_min_entropy_cell = 1e9;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<real,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }
            if(pde_type == PDE_enum::euler && nstate == dim + 2){
                real density = soln_state[0];
                real pressure = euler_physics->compute_pressure(soln_state);
               // real entropy_cell = euler_physics->compute_entropy(density, pressure);
                real entropy_cell = euler_physics->compute_entropy(density, pressure) * density;
                if(entropy_cell < local_min_entropy_cell)
                    local_min_entropy_cell = entropy_cell;
            }
            if(pde_type == PDE_enum::burgers_inviscid){
                real entropy_cell = 0.5 * soln_state[0] * soln_state[0];
                if(entropy_cell < local_min_entropy_cell)
                    local_min_entropy_cell = entropy_cell;
            }
        }
        local_min_entropy(cell_index) = local_min_entropy_cell;
        cell_index++;
    }
}

template <int dim, int nstate, typename real>
void SlopeLimiter<dim, nstate, real>::write_limited_solution(
    dealii::LinearAlgebra::distributed::Vector<double>& solution,
    const std::array<std::vector<real>, nstate>& soln_coeff,
    const unsigned int                                      n_shape_fns,
    const std::vector<dealii::types::global_dof_index>& current_dofs_indices)
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    // Write limited solution dofs to the global solution vector.
    for (int istate = 0; istate < nstate; istate++) {
        for (unsigned int ishape = 0; ishape < n_shape_fns; ++ishape) {
            const unsigned int idof = istate * n_shape_fns + ishape;
            solution[current_dofs_indices[idof]] = soln_coeff[istate][ishape]; //

            // Verify that positivity of density is preserved after application of theta2 limiter
            if(pde_type == PDE_enum::euler && nstate == dim + 2){
                if (istate == 0 && solution[current_dofs_indices[idof]] < 0) {
                    std::cout << "Error: Density is a negative value - Aborting... " << std::endl << solution[current_dofs_indices[idof]] << std::endl << std::flush;
                    std::abort();
                }
            }
        }
    }
}

template <int dim, int nstate, typename real>
void SlopeLimiter<dim, nstate, real>::limit(
    dealii::LinearAlgebra::distributed::Vector<double>&     solution,
    const dealii::DoFHandler<dim>&                          dof_handler,
    const dealii::hp::FECollection<dim>&                    fe_collection,
    const dealii::hp::QCollection<dim>&                     volume_quadrature_collection,
    const unsigned int                                      grid_degree,
    const unsigned int                                      max_degree,
    const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
    const dealii::hp::FECollection<1>                       oneD_fe_collection_leg,
    const dealii::hp::QCollection<1>                        oneD_quadrature_collection,
    double dt)
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    // If use_tvb_limiter is true, apply TVB limiter before applying positivity-preserving limiter
    if (this->all_parameters->limiter_param.use_tvb_limiter == true)
        this->tvbLimiter->limit(solution, dof_handler, fe_collection, volume_quadrature_collection, grid_degree, max_degree, oneD_fe_collection_1state, oneD_fe_collection_1state, oneD_quadrature_collection, dt);

    //first limit for positivity of density
    if(pde_type == PDE_enum::euler && nstate == dim + 2){
        this->posdensity_limiter->limit(solution, dof_handler, fe_collection, volume_quadrature_collection, grid_degree, max_degree, oneD_fe_collection_1state, oneD_fe_collection_1state, oneD_quadrature_collection, dt);
    }

    //create 1D solution polynomial basis functions and corresponding projection operator
    //to interpolate the solution to the quadrature nodes, and to project it back to the
    //modal coefficients.
    const unsigned int init_grid_degree = grid_degree;

    // Constructor for the operators
    OPERATOR::basis_functions<dim, 2 * dim, real> soln_basis(1, max_degree, init_grid_degree);
    OPERATOR::vol_projection_operator<dim, 2 * dim, real> soln_basis_projection_oper(1, max_degree, init_grid_degree);
    OPERATOR::basis_functions<dim, 2 * dim, real> leg_basis(1, max_degree, init_grid_degree);
    OPERATOR::vol_projection_operator<dim, 2 * dim, real> leg_basis_projection_oper(1, max_degree, init_grid_degree);

    // Build the oneD operator to perform interpolation/projection
    soln_basis.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quadrature_collection[max_degree]);
   // soln_basis.build_1D_surface_operator(oneD_fe_collection_1state[max_degree], oneD_face_quadrature);
    soln_basis_projection_oper.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_fe_collection_1state[max_degree], oneD_quadrature_collection[max_degree]);
    leg_basis.build_1D_volume_operator(oneD_fe_collection_leg[max_degree], oneD_quadrature_collection[max_degree]);
    leg_basis_projection_oper.build_1D_volume_operator(oneD_fe_collection_leg[max_degree], oneD_fe_collection_leg[max_degree], oneD_quadrature_collection[max_degree]);

    //basis function 1D at the midpoint
    //the midpoint is the cell center
    const unsigned int n_shape_fns_1D = (max_degree + 1);
    dealii::FullMatrix<real> midpt_basis_1D(1,n_shape_fns_1D);
    dealii::FullMatrix<real> midpt_deriv_basis_1D(1,n_shape_fns_1D);
    const dealii::Point<1> midpt(0.5);//reference element from [0,1]
    for(unsigned int idof=0; idof<n_shape_fns_1D; idof++){
        const int istate = oneD_fe_collection_1state[max_degree].system_to_component_index(idof).first;
        //1D basis fn
        midpt_basis_1D[0][idof] = oneD_fe_collection_1state[max_degree].shape_value_component(idof,midpt,istate);
        //1D derivative basis
        midpt_deriv_basis_1D[0][idof] = oneD_fe_collection_1state[max_degree].shape_grad_component(idof,midpt,istate)[0];
    }
//    const unsigned int n_shape_fns_1D = poly_degree + 1;
//    const unsigned int n_mdpts = pow(n_shape_fns_1D, dim-1);

//    const unsigned int n_face_quad_pts = this->face_quadrature_collection[poly_degree].size();
    /*loop over cells and compute limiter 
    *   then apply the limiter
    */
    for (auto soln_cell : dof_handler.active_cell_iterators()) {
        if (!soln_cell->is_locally_owned()) continue;

        std::vector<dealii::types::global_dof_index> current_dofs_indices;
        // Current reference element related to this physical cell
        const int i_fele = soln_cell->active_fe_index();
        const dealii::FESystem<dim, dim>& current_fe_ref = fe_collection[i_fele];
        const int poly_degree = current_fe_ref.tensor_degree();

        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

        // Obtain the mapping from local dof indices to global dof indices
        current_dofs_indices.resize(n_dofs_curr_cell);
        soln_cell->get_dof_indices(current_dofs_indices);

        // Extract the local solution dofs in the cell from the global solution dofs
        std::array<std::vector<real>, nstate> soln_coeff;

        const unsigned int n_shape_fns = n_dofs_curr_cell / nstate;
        for (unsigned int istate = 0; istate < nstate; ++istate) {
            soln_coeff[istate].resize(n_shape_fns);
        }

        for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
            const unsigned int istate = fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = fe_collection[poly_degree].system_to_component_index(idof).second;
            soln_coeff[istate][ishape] = solution[current_dofs_indices[idof]]; //
        }

        const unsigned int n_quad_pts = volume_quadrature_collection[poly_degree].size();
       // const std::vector<real>& quad_weights = volume_quadrature_collection[poly_degree].get_weights();
        std::array<std::vector<real>, nstate> soln_at_q;

        // Interpolate solution dofs to quadrature pts.
        for (int istate = 0; istate < nstate; istate++) {
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }

        std::array<std::vector<real>, nstate> soln_coeff_leg;
        std::array<std::vector<real>, nstate> soln_coeff_leg_lim;
        // get the Legendre coeff of interior cell.
        for (int istate = 0; istate < nstate; istate++) {
            soln_coeff_leg[istate].resize(n_shape_fns);
            soln_coeff_leg_lim[istate].resize(n_shape_fns);
            leg_basis_projection_oper.matrix_vector_mult_1D(soln_at_q[istate], soln_coeff_leg[istate],
                                                            leg_basis_projection_oper.oneD_vol_operator);
        }

        //get the solution values and gradient at the cell center (midpt)
        std::array<std::vector<real>,nstate> soln_mdpt;
        std::array<std::array<std::vector<real>,dim>,nstate> soln_mdpt_grad;
        for(int istate=0; istate<nstate; istate++){
            soln_mdpt[istate].resize(1);
            soln_basis.matrix_vector_mult(soln_coeff[istate], soln_mdpt[istate],
                                          midpt_basis_1D,
                                          midpt_basis_1D,
                                          midpt_basis_1D);
            for(int idim=0; idim<dim; idim++){
                soln_mdpt_grad[istate][idim].resize(1);
                if(idim==0)
                    soln_basis.matrix_vector_mult(soln_coeff[istate], soln_mdpt_grad[istate][idim],
                                                  midpt_deriv_basis_1D,
                                                  midpt_basis_1D,
                                                  midpt_basis_1D);
                if(idim==1)
                    soln_basis.matrix_vector_mult(soln_coeff[istate], soln_mdpt_grad[istate][idim],
                                                  midpt_basis_1D,
                                                  midpt_deriv_basis_1D,
                                                  midpt_basis_1D);
                if(idim==2)
                    soln_basis.matrix_vector_mult(soln_coeff[istate], soln_mdpt_grad[istate][idim],
                                                  midpt_basis_1D,
                                                  midpt_basis_1D,
                                                  midpt_deriv_basis_1D);
            }
        }
            //value of solution midpoints in each phys direction
      //  for(int idim=0; idim<dim; idim++){
      //      for(int istate=0; istate<nstate; istate++){
               // soln_mdpt[idim][istate].resize(n_mdpts);
               // if(idim==0)
               //     soln_basis.matrix_vector_mult(soln_coeff[istate], soln_mdpt[idim][istate]
               //                                   midpt_basis_1D,
               //                                   soln_basis.oneD_vol_operator,
               //                                   soln_basis.oneD_vol_operator);
               // if(idim==1)
               //     soln_basis.matrix_vector_mult(soln_coeff[istate], soln_mdpt[idim][istate]
               //                                   soln_basis.oneD_vol_operator,
               //                                   midpt_basis_1D,
               //                                   soln_basis.oneD_vol_operator);
               // if(idim==2)
               //     soln_basis.matrix_vector_mult(soln_coeff[istate], soln_mdpt[idim][istate]
               //                                   soln_basis.oneD_vol_operator,
               //                                   soln_basis.oneD_vol_operator,
               //                                   midpt_basis_1D);
           // }
            //soln midpt gradients
         //   for(int jdim=0; jdim<dim; jdim++){
         //       for(int istate=0; istate<nstate; istate++){
         //          // soln_mdpt_grad[idim][istate].resize(n_mdpts);
         //           if(idim==0)
         //               soln_basis.matrix_vector_mult(soln_coeff[istate], soln_mdpt[idim][istate]
         //                                             midpt_basis_1D,
         //                                             soln_basis.oneD_vol_operator,
         //                                             soln_basis.oneD_vol_operator);
         //           if(idim==1)
         //               soln_basis.matrix_vector_mult(soln_coeff[istate], soln_mdpt[idim][istate]
         //                                             soln_basis.oneD_vol_operator,
         //                                             midpt_basis_1D,
         //                                             soln_basis.oneD_vol_operator);
         //           if(idim==2)
         //               soln_basis.matrix_vector_mult(soln_coeff[istate], soln_mdpt[idim][istate]
         //                                             soln_basis.oneD_vol_operator,
         //                                             soln_basis.oneD_vol_operator,
         //                                             midpt_basis_1D);
         //       }
         //   }
        //}//

        //get solution coeff neighbours
        const unsigned int n_faces = dealii::GeometryInfo<dim>::faces_per_cell;
        std::array<std::array<std::vector<real>, nstate>,6> leg_neigh_coeff;
        //std::array<std::array<std::vector<real>, nstate>,6> soln_at_surf_q;
        std::array<std::array<std::vector<real>, nstate>,6> neigh_soln_mdpt;
        std::array<std::array<std::array<std::vector<real>,dim>, nstate>,6> neigh_soln_mdpt_grad;
       // std::array<std::array<std::vector<real>, nstate>,6> neigh_soln_at_surf_q;
      //  std::array<std::array<std::array<std::vector<real>, nstate>,6>,dim> neigh_soln_mdpt;
      //  std::array<std::array<std::vector<real>, nstate>,6> neigh_soln_at_surf_q;
        //loop over faces and get neighbour soln coeff
        for (unsigned int iface=0; iface < n_faces; ++iface) {
            auto current_face = soln_cell->face(iface);
            if ((current_face->at_boundary() && !soln_cell->has_periodic_neighbor(iface)))
                continue;
            const auto neighbor_cell = (current_face->at_boundary() && soln_cell->has_periodic_neighbor(iface)) ? soln_cell->periodic_neighbor(iface) : soln_cell->neighbor_or_periodic_neighbor(iface);

           // const unsigned int neighbor_iface = (current_face->at_boundary() && soln_cell->has_periodic_neighbor(iface)) ? soln_cell->periodic_neighbor_of_periodic_neighbor(iface) : soln_cell->neighbor_of_neighbor(iface);
            const unsigned int n_dofs_neigh_cell = fe_collection[neighbor_cell->active_fe_index()].n_dofs_per_cell();
            std::vector<dealii::types::global_dof_index> neighbor_dofs_indices;
            neighbor_dofs_indices.resize(n_dofs_neigh_cell);
            neighbor_cell->get_dof_indices (neighbor_dofs_indices);
            const int poly_degree_ext = neighbor_cell->active_fe_index();
            const unsigned int n_dofs_ext = fe_collection[poly_degree_ext].dofs_per_cell;
            const unsigned int n_shape_fns_ext = n_dofs_ext / nstate;
            std::array<std::vector<real>, nstate> soln_neigh_coeff;
            for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
                const unsigned int istate = fe_collection[poly_degree_ext].system_to_component_index(idof).first;
                const unsigned int ishape = fe_collection[poly_degree_ext].system_to_component_index(idof).second;
                if(ishape == 0){
                    soln_neigh_coeff[istate].resize(n_shape_fns_ext);
                }
                soln_neigh_coeff[istate][ishape] = solution(neighbor_dofs_indices[idof]);
            }
            std::array<std::vector<real>, nstate> soln_neigh_q;
            // Interpolate solution dofs to quadrature pts.
            for (int istate = 0; istate < nstate; istate++) {
                soln_neigh_q[istate].resize(n_quad_pts);
                soln_basis.matrix_vector_mult_1D(soln_neigh_coeff[istate], soln_neigh_q[istate],
                                                 soln_basis.oneD_vol_operator);
            }
             
            // Interpolate solution dofs to quadrature pts.
            for (int istate = 0; istate < nstate; istate++) {
                leg_neigh_coeff[iface][istate].resize(n_shape_fns_ext);
                leg_basis_projection_oper.matrix_vector_mult_1D(soln_neigh_q[istate], leg_neigh_coeff[iface][istate],
                                                                leg_basis_projection_oper.oneD_vol_operator);
            }
            for(int istate=0; istate<nstate; istate++){
                neigh_soln_mdpt[iface][istate].resize(1);
                soln_basis.matrix_vector_mult(soln_neigh_coeff[istate], neigh_soln_mdpt[iface][istate],
                                              midpt_basis_1D,
                                              midpt_basis_1D,
                                              midpt_basis_1D);
                for(int idim=0; idim<dim; idim++){
                    neigh_soln_mdpt_grad[iface][istate][idim].resize(1);
                    if(idim==0)
                        soln_basis.matrix_vector_mult(soln_neigh_coeff[istate], neigh_soln_mdpt_grad[iface][istate][idim],
                                                      midpt_deriv_basis_1D,
                                                      midpt_basis_1D,
                                                      midpt_basis_1D);
                    if(idim==1)
                        soln_basis.matrix_vector_mult(soln_neigh_coeff[istate], neigh_soln_mdpt_grad[iface][istate][idim],
                                                      midpt_basis_1D,
                                                      midpt_deriv_basis_1D,
                                                      midpt_basis_1D);
                    if(idim==2)
                        soln_basis.matrix_vector_mult(soln_neigh_coeff[istate], neigh_soln_mdpt_grad[iface][istate][idim],
                                                      midpt_basis_1D,
                                                      midpt_basis_1D,
                                                      midpt_deriv_basis_1D);
                }
            }
        }//end face loop
       //     //interpolate solution to the face quad pts
       //     for (int istate = 0; istate < nstate; istate++) {
       //         soln_at_face_q[iface][istate].resize(n_face_quad_pts);
       //         soln_basis.matrix_vector_mult_surface_1D(iface,
       //                                                  soln_coeff[istate],
       //                                                  soln_at_face_q[iface][istate],
       //                                                  soln_basis.oneD_surf_operator,
       //                                                  soln_basis.oneD_vol_operator);
       //     }
       //     //get the neighbour soln midpts
       //     //get the solution at the midpoints and tensor product for all 3 directions
       //     for(int idim=0; idim<dim; idim++){
       //         for(int istate=0; istate<nstate; istate++){
       //             neigh_soln_mdpt[idim][iface][istate].resize(n_mdpts);
       //             if(idim==0)
       //                 soln_basis.matrix_vector_mult(soln_neigh_coeff[istate], neigh_soln_mdpt[idim][iface][istate]
       //                                               midpt_basis_1D,
       //                                               soln_basis.oneD_vol_operator,
       //                                               soln_basis.oneD_vol_operator);
       //             if(idim==1)
       //                 soln_basis.matrix_vector_mult(soln_neigh_coeff[istate], neigh_soln_mdpt[idim][iface][istate]
       //                                               soln_basis.oneD_vol_operator,
       //                                               midpt_basis_1D,
       //                                               soln_basis.oneD_vol_operator);
       //             if(idim==2)
       //                 soln_basis.matrix_vector_mult(soln_neigh_coeff[istate], neigh_soln_mdpt[idim][iface][istate]
       //                                               soln_basis.oneD_vol_operator,
       //                                               soln_basis.oneD_vol_operator,
       //                                               midpt_basis_1D);
       //         }
       //     }
       //         for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){ 
       //             a12[iquad] = 2.0 * soln_mdpt[idim][istate][iquad] - soln_at_face_q[other_face][istate][iquad];//assuming the index order the same at the opposite face for the recontrcution
       //             a13[iquad] = 2.0 * neigh_soln_mdpt[idim][istate][iquad] - neigh_soln_at_face_q[neigh_other_face][istate][iquad];//assuming the index order the same at the opposite face for the recontrcution
       //         }

        
      //  std::array<std::array<std::vector<double>,nstate>,6> u_lim_ext;
      //  for(int istate=0; istate<nstate; istate++){
      //      for(int iface=0; iface<n_faces; iface++){
      //          u_lim_ext[iface][istate].resize(n_face_quad_pts);
      //          for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
      //              u_lim_ext[iface][istate][iquad] = abs(leg_neigh_coeff[iface][istate][0] - soln_coeff_leg[istate][0]);
      //          }
      //      }
      //  }


        //extended range
        std::array<real,nstate> u_lim; 
        for(int istate=0; istate<nstate; istate++){
            u_lim[istate] = 100000;
            for(unsigned int iface=0; iface<n_faces; iface++){
                //get neighbour facees
                auto current_face = soln_cell->face(iface);
                if ((current_face->at_boundary() && !soln_cell->has_periodic_neighbor(iface)))
                    continue;
                real a11 = 0.5 * (soln_coeff_leg[istate][0] + leg_neigh_coeff[iface][istate][0]);
              //  const auto neighbor_cell = (current_face->at_boundary() && soln_cell->has_periodic_neighbor(iface)) ? soln_cell->periodic_neighbor(iface) : soln_cell->neighbor_or_periodic_neighbor(iface);
                 
                const unsigned int neighbor_iface = (current_face->at_boundary() && soln_cell->has_periodic_neighbor(iface)) ? soln_cell->periodic_neighbor_of_periodic_neighbor(iface) : soln_cell->neighbor_of_neighbor(iface);
                real grad_dot_one_int = 0.0;
                real grad_dot_one_ext = 0.0;
                for(int idim=0; idim<dim; idim++){
                    grad_dot_one_int += soln_mdpt_grad[istate][idim][0] * 1.0;
                    grad_dot_one_ext += neigh_soln_mdpt_grad[iface][istate][idim][0] * 1.0;
                }
                double sign = (iface == 1) ? 1.0 : -1.0;
                double sign_ext = (neighbor_iface == 1) ? 1.0 : -1.0;
                real a12 = soln_mdpt[istate][0] + sign * grad_dot_one_int * 0.5;
                real a13 = neigh_soln_mdpt[iface][istate][0] + sign_ext * grad_dot_one_ext * 0.5;
                real sign_a12 = (a12 > 0.0) ? 1.0 : -1.0;
                real sign_a13 = (a13 > 0.0) ? 1.0 : -1.0;
                real min_abs = (abs(a12) < abs(a13)) ? abs(a12) : abs(a13);
                real a14 = a11 + 0.5 * (sign_a12 + sign_a13) * min_abs;
                real dif_avg = abs(leg_neigh_coeff[iface][istate][0]-soln_coeff_leg[istate][0]);
                real lim = (dif_avg < a14 ) ? a14 : dif_avg;//take max of 2
                if(lim < u_lim[istate])
                    u_lim[istate] = lim;
            }
        }


        //take the max between the extended range and the limited value before
        //apply the limiter
        for(int istate=0; istate<nstate; istate++){
            soln_coeff_leg_lim[istate][0] = soln_coeff_leg[istate][0];
            real u_lim_mode = u_lim[istate];
            for(unsigned int ishape=1; ishape<n_shape_fns; ishape++){
                const double sign = soln_coeff_leg[istate][ishape] >= 0.0 ? 1.0 : -1.0;
                const real min_val = (soln_coeff_leg[istate][ishape] < u_lim_mode) ? soln_coeff_leg[istate][ishape] : u_lim_mode;
                soln_coeff_leg_lim[istate][ishape] = sign * min_val;
                u_lim_mode = u_lim_mode - abs(soln_coeff_leg_lim[istate][ishape]);
            }
        }
        
        //interpolate/project back the limited solution
        for(int istate=0; istate<nstate; istate++){
            leg_basis.matrix_vector_mult_1D(soln_coeff_leg_lim[istate], soln_at_q[istate],
                                             leg_basis.oneD_vol_operator);
            soln_basis_projection_oper.matrix_vector_mult_1D(soln_at_q[istate], soln_coeff[istate],
                                             soln_basis_projection_oper.oneD_vol_operator);

        }
        

    }
}

template class SlopeLimiter <PHILIP_DIM, 1, double>;
template class SlopeLimiter <PHILIP_DIM, 2, double>;
template class SlopeLimiter <PHILIP_DIM, 3, double>;
template class SlopeLimiter <PHILIP_DIM, 4, double>;
template class SlopeLimiter <PHILIP_DIM, 5, double>;
template class SlopeLimiter <PHILIP_DIM, 6, double>;
} // PHiLiP namespace
