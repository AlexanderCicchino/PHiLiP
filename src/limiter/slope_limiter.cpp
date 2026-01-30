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


//    if (pde_type == PDE_enum::euler && nstate == dim + 2) {
//        std::shared_ptr< ManufacturedSolutionFunction<dim, real> >  manufactured_solution_function
//            = ManufacturedSolutionFactory<dim, real>::create_ManufacturedSolution(parameters_input, nstate);
//        euler_physics = std::make_shared < Physics::Euler<dim, nstate, real> >(
//            parameters_input,
//            parameters_input->euler_param.ref_length,
//            parameters_input->euler_param.gamma_gas,
//            parameters_input->euler_param.mach_inf,
//            parameters_input->euler_param.angle_of_attack,
//            parameters_input->euler_param.side_slip_angle,
//            manufactured_solution_function,
//            parameters_input->two_point_num_flux_type);
//    }
//    else if (pde_type == PDE_enum::burgers_inviscid && dim == 1) {
//        std::shared_ptr< ManufacturedSolutionFunction<1, real> >  manufactured_solution_function
//            = ManufacturedSolutionFactory<1, real>::create_ManufacturedSolution(parameters_input, 1);
//        burgers_physics = std::make_shared < Physics::Burgers<1, 1, real> >(
//                parameters_input,
//                parameters_input->burgers_param.diffusion_coefficient,
//                true, false,
//                parameters_input->manufactured_convergence_study_param.manufactured_solution_param.diffusion_tensor, 
//                manufactured_solution_function,
//                parameters_input->test_type);
//    }
//    else {
//        std::cout << "Error: Slope Limiter can only be applied for pde_type==euler or burgers_inviscid" << std::endl;
//        std::abort();
//    }
    this->posdensity_limiter = std::make_shared < PositivityPreservingLimiter<dim, nstate, real> >(parameters_input);
}

//template <int dim, int nstate, typename real>
//void SlopeLimiter<dim, nstate, real>::set_cell_min_entropy(
//    dealii::LinearAlgebra::distributed::Vector<double>&     solution,
//    const dealii::DoFHandler<dim>&                          dof_handler,
//    const dealii::hp::FECollection<dim>&                    fe_collection,
//    const dealii::hp::QCollection<dim>&                     volume_quadrature_collection,
//    const unsigned int                                      grid_degree,
//    const unsigned int                                      max_degree,
//    const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
//    const dealii::hp::QCollection<1>                        oneD_quadrature_collection)
//{
//    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
//
//    unsigned int n_cells_local=0;
//    for (auto soln_cell : dof_handler.active_cell_iterators()) {
//        if (!soln_cell->is_locally_owned()) continue;
//        n_cells_local++;
//    }
//    local_min_entropy.reinit(n_cells_local);
//
//
//    const unsigned int init_grid_degree = grid_degree;
//
//    // Constructor for the operators
//    OPERATOR::basis_functions<dim, 2 * dim, real> soln_basis(1, max_degree, init_grid_degree);
//    OPERATOR::vol_projection_operator<dim, 2 * dim, real> soln_basis_projection_oper(1, max_degree, init_grid_degree);
//
//    // Build the oneD operator to perform interpolation/projection
//    soln_basis.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quadrature_collection[max_degree]);
//    soln_basis_projection_oper.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quadrature_collection[max_degree]);
//
//    /*loop over cells and compute limiter for min enetorpy principle
//    *   then apply the limiter
//    */
//    unsigned int cell_index = 0;
//    for (auto soln_cell : dof_handler.active_cell_iterators()) {
//        if (!soln_cell->is_locally_owned()) continue;
//
//        std::vector<dealii::types::global_dof_index> current_dofs_indices;
//        // Current reference element related to this physical cell
//        const int i_fele = soln_cell->active_fe_index();
//        const dealii::FESystem<dim, dim>& current_fe_ref = fe_collection[i_fele];
//        const int poly_degree = current_fe_ref.tensor_degree();
//
//        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();
//
//        // Obtain the mapping from local dof indices to global dof indices
//        current_dofs_indices.resize(n_dofs_curr_cell);
//        soln_cell->get_dof_indices(current_dofs_indices);
//
//        // Extract the local solution dofs in the cell from the global solution dofs
//        std::array<std::vector<real>, nstate> soln_coeff;
//
//        const unsigned int n_shape_fns = n_dofs_curr_cell / nstate;
//        for (unsigned int istate = 0; istate < nstate; ++istate) {
//            soln_coeff[istate].resize(n_shape_fns);
//        }
//
//        for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
//            const unsigned int istate = fe_collection[poly_degree].system_to_component_index(idof).first;
//            const unsigned int ishape = fe_collection[poly_degree].system_to_component_index(idof).second;
//            soln_coeff[istate][ishape] = solution[current_dofs_indices[idof]]; //
//        }
//
//        const unsigned int n_quad_pts = volume_quadrature_collection[poly_degree].size();
//        std::array<std::vector<real>, nstate> soln_at_q;
//
//        // Interpolate solution dofs to quadrature pts.
//        for (int istate = 0; istate < nstate; istate++) {
//            soln_at_q[istate].resize(n_quad_pts);
//            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
//                                             soln_basis.oneD_vol_operator);
//        }
//
//        // Obtain solution cell average
//        real local_min_entropy_cell = 1e9;
//        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
//            std::array<real,nstate> soln_state;
//            for(int istate=0; istate<nstate; istate++){
//                soln_state[istate] = soln_at_q[istate][iquad];
//            }
//            if(pde_type == PDE_enum::euler && nstate == dim + 2){
//                real density = soln_state[0];
//                real pressure = euler_physics->compute_pressure(soln_state);
//               // real entropy_cell = euler_physics->compute_entropy(density, pressure);
//                real entropy_cell = euler_physics->compute_entropy(density, pressure) * density;
//                if(entropy_cell < local_min_entropy_cell)
//                    local_min_entropy_cell = entropy_cell;
//            }
//            if(pde_type == PDE_enum::burgers_inviscid){
//                real entropy_cell = 0.5 * soln_state[0] * soln_state[0];
//                if(entropy_cell < local_min_entropy_cell)
//                    local_min_entropy_cell = entropy_cell;
//            }
//        }
//        local_min_entropy(cell_index) = local_min_entropy_cell;
//        cell_index++;
//    }
//}

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
void SlopeLimiter<dim, nstate, real>::minmod(
    const real &a,
    const real &b,
    real &val)
{
    real sign_a = (a>=0.0) ? 1.0 : -1.0;
    real sign_b = (b>=0.0) ? 1.0 : -1.0;
    real min_abs = (abs(a) < abs(b)) ? abs(a) : abs(b);
    val = 0.5 * (sign_a + sign_b) * min_abs;

}
template <int dim, int nstate, typename real>
void SlopeLimiter<dim, nstate, real>::eigenvect_cons_to_char_var(
    const std::array<real,nstate> &/*soln*/,
    const std::array<std::array<std::array<real,nstate>,nstate>,dim> &right_eig,
    std::array<std::array<std::array<real,nstate>,nstate>,dim> &eigenvect)
{
    for(int idim=0; idim<dim; idim++){
        dealii::FullMatrix<real> l(nstate,nstate);
        dealii::FullMatrix<real> r(nstate,nstate);
        for(int i=0; i<nstate; i++){
            for(int j=0; j<nstate; j++){
                r[i][j] = right_eig[idim][i][j];
            }
        }
        l.invert(r);
        for(int i=0; i<nstate; i++){
            for(int j=0; j<nstate; j++){
                eigenvect[idim][i][j] = l[i][j];
            }
        }
    }
}
template <int dim, int nstate, typename real>
void SlopeLimiter<dim, nstate, real>::eigenvect_char_to_cons_var(
    const std::array<real,nstate> &soln,
    std::array<std::array<std::array<real,nstate>,nstate>,dim> &eigenvect)
{
    const real density = soln[0];
    dealii::Tensor<1,dim,real> vel;
    real vel2 = 0.0;
    for(int idim=0; idim<dim; idim++){
        vel[idim] = soln[idim+1]/soln[0];
        vel2 += vel[idim] * vel[idim];
    }
    const real pressure = (0.4) * (soln[nstate-1] - 0.5 * density * vel2);
    const real specific_total_enthalpy = soln[nstate-1]/density + pressure/density;
    const real speed_sound = sqrt(pressure*1.4/density);
    const real scale = 2.0*speed_sound;
    if constexpr(dim==1){
       // eigenvect[0][0][0] = 1.0;
        eigenvect[0][0][0] = 1.0/scale;
        eigenvect[0][0][1] = 1.0;
       // eigenvect[0][0][2] = 1.0;
        eigenvect[0][0][2] = 1.0/scale;
       // eigenvect[0][1][0] = vel[0] - speed_sound;
        eigenvect[0][1][0] = (vel[0] - speed_sound)/scale;
        eigenvect[0][1][1] = vel[0];
       // eigenvect[0][1][2] = vel[0] + speed_sound;
        eigenvect[0][1][2] = (vel[0] + speed_sound)/scale;
       // eigenvect[0][2][0] = specific_total_enthalpy - vel[0] * speed_sound;
        eigenvect[0][2][0] = (specific_total_enthalpy - vel[0] * speed_sound)/scale;
        eigenvect[0][2][1] = 0.5 * vel2;
       // eigenvect[0][2][2] = specific_total_enthalpy + vel[0] * speed_sound;
        eigenvect[0][2][2] = (specific_total_enthalpy + vel[0] * speed_sound)/scale;
    }
    if constexpr(dim==2){
        //x-direction
        //row 1
        eigenvect[0][0][0] = 1.0/scale;
        eigenvect[0][0][1] = 0.0;
        eigenvect[0][0][2] = 1.0;
        eigenvect[0][0][3] = 1.0/scale;
        //row 2
        eigenvect[0][1][0] = (vel[0] - speed_sound)/scale;
        eigenvect[0][1][1] = 0.0;
        eigenvect[0][1][2] = vel[0];
        eigenvect[0][1][3] = (vel[0] + speed_sound)/scale;
        //row 3
        eigenvect[0][2][0] = (vel[1])/scale;
        eigenvect[0][2][1] = 1.0;
        eigenvect[0][2][2] = vel[1];
        eigenvect[0][2][3] = (vel[1])/scale;
        //row 4
        eigenvect[0][3][0] = (specific_total_enthalpy - vel[0] * speed_sound)/scale;
        eigenvect[0][3][1] = vel[1];
        eigenvect[0][3][2] = 0.5 * vel2;
        eigenvect[0][3][3] = (specific_total_enthalpy + vel[0] * speed_sound)/scale;
        //y-direction
        //row 1
        eigenvect[1][0][0] = (1.0)/scale;
        eigenvect[1][0][1] = 1.0;
        eigenvect[1][0][2] = 0.0;
        eigenvect[1][0][3] = (1.0)/scale;
        //row 2
        eigenvect[1][1][0] = (vel[0])/scale;
        eigenvect[1][1][1] = vel[0];
        eigenvect[1][1][2] = 1.0;
        eigenvect[1][1][3] = (vel[0])/scale;
        //row 3
        eigenvect[1][2][0] = (vel[1] - speed_sound)/scale;
        eigenvect[1][2][1] = vel[1];
        eigenvect[1][2][2] = 0.0;
        eigenvect[1][2][3] = (vel[1] + speed_sound)/scale;
        //row 4
        eigenvect[1][3][0] = (specific_total_enthalpy - vel[1] * speed_sound)/scale;
        eigenvect[1][3][1] = 0.5 * vel2;
        eigenvect[1][3][2] = vel[0];
        eigenvect[1][3][3] = (specific_total_enthalpy + vel[1] * speed_sound)/scale;
    }
}
template <int dim, int nstate, typename real>
void SlopeLimiter<dim, nstate, real>::transform_cons_to_char_var(
    const std::array<real,nstate> &soln,
    std::array<real,nstate> &char_var)
{
    const real density = soln[0];
    dealii::Tensor<1,dim,real> vel;
    real vel2 = 0.0;
    for(int idim=0; idim<dim; idim++){
        vel[idim] = soln[idim+1]/soln[0];
        vel2 += vel[idim] * vel[idim];
    }
    const real p = (0.4) * (soln[nstate-1] - 0.5 * density * vel2);
   // const real h = soln[nstate-1]/density + p/density;
    const real a = sqrt(p*1.4/density);
    const real a2 = a*a;
    std::array<std::array<real,nstate>,nstate> eigenvect{};
    if constexpr(dim==1){
        eigenvect[0][0] = (0.4)/4.0*vel[0]*vel[0]/(a2) + vel[0]/(2.0*a);
        eigenvect[0][1] =  -0.4/2.0*vel[0]/a2 - 1.0/2.0/a;
        eigenvect[0][2] = 0.4/(2.0*a2);
        eigenvect[1][0] = 1.0 - 0.4/2.0*vel[0]*vel[0]/a2;
        eigenvect[1][1] = 0.4*vel[0]/a2;
        eigenvect[1][2] = -0.4/a2;
        eigenvect[2][0] = 0.4/4.0*vel[0]*vel[0]/a2 - vel[0]/2.0/a;
        eigenvect[2][1] = -0.4/2.0*vel[0]/a2 + 1.0/2.0/a;
        eigenvect[2][2] = 0.4/2.0/a2;
        
    }
    for(int istate=0; istate<nstate; istate++){
        char_var[istate] = 0.0;
        for(int jstate=0; jstate<nstate; jstate++){
            char_var[istate] += eigenvect[istate][jstate] * soln[jstate];
        }
    }

}
template <int dim, int nstate, typename real>
void SlopeLimiter<dim, nstate, real>::transform_char_to_cons_var(
    const std::array<real,nstate> &char_var,
    const std::array<real,nstate> &soln_prev,
    std::array<real,nstate> &soln)
{
    const real density = soln_prev[0];
    dealii::Tensor<1,dim,real> vel;
    real vel2 = 0.0;
    for(int idim=0; idim<dim; idim++){
        vel[idim] = soln_prev[idim+1]/soln_prev[0];
        vel2 += vel[idim] * vel[idim];
    }
    const real pressure = (0.4) * (soln_prev[nstate-1] - 0.5 * density * vel2);
    const real specific_total_enthalpy = soln_prev[nstate-1]/density + pressure/density;
    const real speed_sound = sqrt(pressure*1.4/density);
    std::array<std::array<real,nstate>,nstate> eigenvect{};
    if constexpr(dim==1){
        eigenvect[0][0] = 1.0;
        eigenvect[0][1] = 1.0;
        eigenvect[0][2] = 1.0;
        eigenvect[1][0] = vel[0] - speed_sound;
        eigenvect[1][1] = vel[0];
        eigenvect[1][2] = vel[0] + speed_sound;
        eigenvect[2][0] = specific_total_enthalpy - vel[0] * speed_sound;
        eigenvect[2][1] = 0.5 * vel2;
        eigenvect[2][2] = specific_total_enthalpy + vel[0] * speed_sound;
        
    }
    for(int istate=0; istate<nstate; istate++){
        soln[istate] = 0.0;
        for(int jstate=0; jstate<nstate; jstate++){
            soln[istate] += eigenvect[istate][jstate] * char_var[jstate];
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
    const dealii::hp::FECollection<1>                       /*oneD_fe_collection_leg*/,
    const dealii::hp::QCollection<1>                        oneD_quadrature_collection,
    double dt)
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    // If use_tvb_limiter is true, apply TVB limiter before applying positivity-preserving limiter
    if (this->all_parameters->limiter_param.use_tvb_limiter == true)
        this->tvbLimiter->limit(solution, dof_handler, fe_collection, volume_quadrature_collection, grid_degree, max_degree, oneD_fe_collection_1state, oneD_fe_collection_1state, oneD_quadrature_collection, dt);


    solution_update.reinit(solution);
    //first limit for positivity of density
//    if(pde_type == PDE_enum::euler && nstate == dim + 2){
//        this->posdensity_limiter->limit(solution, dof_handler, fe_collection, volume_quadrature_collection, grid_degree, max_degree, oneD_fe_collection_1state, oneD_fe_collection_1state, oneD_quadrature_collection, dt);
//    }

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
    dealii::QGauss<0> oneD_face_quadrature(max_degree);
    soln_basis.build_1D_surface_operator(oneD_fe_collection_1state[max_degree], oneD_face_quadrature);
    soln_basis_projection_oper.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quadrature_collection[max_degree]);

    /*loop over cells and compute filter 
    *   then apply the filter
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
        const std::vector<real>& quad_weights = volume_quadrature_collection[poly_degree].get_weights();
        std::array<std::vector<real>, nstate> soln_at_q;
        std::array<std::vector<real>, nstate> soln_at_q_new;
        std::array<real,nstate> soln_avg;
        const unsigned int n_face_quad_pts = (poly_degree+1);
        // Interpolate solution dofs to quadrature pts.
        for (int istate = 0; istate < nstate; istate++) {
            soln_at_q[istate].resize(n_quad_pts);
            soln_at_q_new[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
            //get cell average
            soln_avg[istate]=0.0;
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                soln_avg[istate] += soln_at_q[istate][iquad] * quad_weights[iquad];
            }
            //set soln q new interior quad the same
            for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
                for(unsigned int jquad=0; jquad<n_face_quad_pts; jquad++){
                    const unsigned int index = jquad + iquad * n_face_quad_pts;
                    if(iquad ==0 || jquad == 0 || iquad == n_face_quad_pts-1 || jquad == n_face_quad_pts-1){
                        soln_at_q_new[istate][index] = 0.0;
                        continue;
                    }
                    soln_at_q_new[istate][index] = soln_at_q[istate][index];
                }
            }
        }
            //enforce continuous solution at faces
        //get solution coeff neighbours
        const unsigned int n_faces = dealii::GeometryInfo<dim>::faces_per_cell;
        std::array<std::array<std::vector<real>, nstate>,6> leg_neigh_coeff;
        std::array<std::array<std::vector<real>, nstate>,6> neigh_soln_mdpt;
        std::array<std::array<std::array<std::vector<real>,dim>, nstate>,6> neigh_soln_mdpt_grad;
        std::array<std::array<std::array<std::vector<real>, nstate>,6>,dim> neigh_char_coeff_leg;
        //loop over faces and get neighbour soln coeff
        for (unsigned int iface=0; iface < n_faces; ++iface) {
            auto current_face = soln_cell->face(iface);
            if ((current_face->at_boundary() && !soln_cell->has_periodic_neighbor(iface)))
                continue;
            const auto neighbor_cell = (current_face->at_boundary() && soln_cell->has_periodic_neighbor(iface)) ? soln_cell->periodic_neighbor(iface) : soln_cell->neighbor_or_periodic_neighbor(iface);

            const unsigned int neighbor_iface = (current_face->at_boundary() && soln_cell->has_periodic_neighbor(iface)) ? soln_cell->periodic_neighbor_of_periodic_neighbor(iface) : soln_cell->neighbor_of_neighbor(iface);
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
            std::array<std::vector<real>,nstate> neigh_soln_face;
            for (int istate = 0; istate < nstate; istate++) {
                soln_neigh_q[istate].resize(n_quad_pts);
                soln_basis.matrix_vector_mult_1D(soln_neigh_coeff[istate], soln_neigh_q[istate],
                                                 soln_basis.oneD_vol_operator);
                neigh_soln_face[istate].resize(n_face_quad_pts);
                for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
                    if(neighbor_iface==0){//left face
                        const unsigned int index = iquad * n_face_quad_pts;
                        neigh_soln_face[istate][iquad] = soln_neigh_q[istate][index];
                    }
                    if(neighbor_iface==1){//right face
                        const unsigned int index = iquad * n_face_quad_pts + n_face_quad_pts-1;
                        neigh_soln_face[istate][iquad] = soln_neigh_q[istate][index];
                    }
                    if(neighbor_iface==2){//bottom face
                        const unsigned int index = iquad;
                        neigh_soln_face[istate][iquad] = soln_neigh_q[istate][index];
                    }
                    if(neighbor_iface==3){//top face
                        const unsigned int index = iquad + (n_face_quad_pts-1) * n_face_quad_pts;
                        neigh_soln_face[istate][iquad] = soln_neigh_q[istate][index];
                    }
                }
                for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
                    real scale = 0.5;
                    real scale2 = 1.0;
                    if(iquad == 0 || iquad == n_face_quad_pts - 1){
                        if(iface==0 && neighbor_iface ==1 ){//left face
                            const unsigned int index = iquad * n_face_quad_pts;
                            soln_at_q_new[istate][index] = neigh_soln_face[istate][iquad];
                        }
                        if(iface==1 && neighbor_iface == 0){//right face
                            const unsigned int index = iquad * n_face_quad_pts + n_face_quad_pts-1;
                            soln_at_q_new[istate][index] = soln_at_q[istate][index];
                        }
                        if(iface==2 && neighbor_iface==3){//bottom face
                            const unsigned int index = iquad;
                            soln_at_q_new[istate][index] = neigh_soln_face[istate][iquad];
                        }
                        if(iface==3 && neighbor_iface==2){//top face
                            const unsigned int index = iquad + (n_face_quad_pts-1) * n_face_quad_pts;
                            soln_at_q_new[istate][index] = soln_at_q[istate][index];
                        }
                    }
                    else{
                        if(iface==0){//left face
                            const unsigned int index = iquad * n_face_quad_pts;
                            soln_at_q_new[istate][index] += scale * (neigh_soln_face[istate][iquad] + scale2 * soln_at_q[istate][index]);
                        }
                        if(iface==1){//right face
                            const unsigned int index = iquad * n_face_quad_pts + n_face_quad_pts-1;
                            soln_at_q_new[istate][index] += scale * (neigh_soln_face[istate][iquad] + scale2 * soln_at_q[istate][index]);
                        }
                        if(iface==2){//bottom face
                            const unsigned int index = iquad;
                            soln_at_q_new[istate][index] += scale * (neigh_soln_face[istate][iquad] + scale2 * soln_at_q[istate][index]);
                        }
                        if(iface==3){//top face
                            const unsigned int index = iquad + (n_face_quad_pts-1) * n_face_quad_pts;
                            soln_at_q_new[istate][index] += scale * (neigh_soln_face[istate][iquad] + scale2 * soln_at_q[istate][index]);
                        }
                    }
                }
            }

        }//end face loop
        //get soln q new avg
        std::array<real,nstate> soln_q_new_avg;
        for(int istate=0; istate<nstate; istate++){
            soln_q_new_avg[istate] = 0.0;
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                soln_q_new_avg[istate] += soln_at_q_new[istate][iquad] * quad_weights[iquad];
            }
            //correct the average
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                soln_at_q_new[istate][iquad] = soln_at_q_new[istate][iquad] + soln_avg[istate] - soln_q_new_avg[istate];
            }
        }

        for (int istate = 0; istate < nstate; istate++) {
           // soln_basis_projection_oper.matrix_vector_mult_1D(soln_at_q[istate], soln_coeff[istate],
            soln_basis_projection_oper.matrix_vector_mult_1D(soln_at_q_new[istate], soln_coeff[istate],
                                             soln_basis_projection_oper.oneD_vol_operator);
        }
        
       // write_limited_solution(solution, soln_coeff, n_shape_fns, current_dofs_indices);
        write_limited_solution(solution_update, soln_coeff, n_shape_fns, current_dofs_indices);

    }
    solution = solution_update;
}

template class SlopeLimiter <PHILIP_DIM, PHILIP_DIM+2, double>;
//template class SlopeLimiter <PHILIP_DIM, 1, double>;
//template class SlopeLimiter <PHILIP_DIM, 2, double>;
//template class SlopeLimiter <PHILIP_DIM, 3, double>;
//template class SlopeLimiter <PHILIP_DIM, 4, double>;
//template class SlopeLimiter <PHILIP_DIM, 5, double>;
//template class SlopeLimiter <PHILIP_DIM, 6, double>;
} // PHiLiP namespace

