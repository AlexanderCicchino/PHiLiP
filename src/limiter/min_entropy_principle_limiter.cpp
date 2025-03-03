#include "positivity_preserving_limiter.h"
#include "tvb_limiter.h"
#include "min_entropy_principle_limiter.h"

namespace PHiLiP {
/**********************************
*
* Positivity Preserving Limiter Class
*
**********************************/
// Constructor
template <int dim, int nstate, typename real>
MinEntropyPrincipleLimiter<dim, nstate, real>::MinEntropyPrincipleLimiter(
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
        std::cout << "Error: Minimum Entropy Principle Limiter can only be applied for pde_type==euler or burgers_inviscid" << std::endl;
        std::abort();
    }
    this->posdensity_limiter = std::make_shared < PositivityPreservingLimiter<dim, nstate, real> >(parameters_input);
}

template <int dim, int nstate, typename real>
void MinEntropyPrincipleLimiter<dim, nstate, real>::set_cell_min_entropy(
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
    soln_basis_projection_oper.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quadrature_collection[max_degree]);

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
void MinEntropyPrincipleLimiter<dim, nstate, real>::write_limited_solution(
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
void MinEntropyPrincipleLimiter<dim, nstate, real>::limit(
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
    // If use_tvb_limiter is true, apply TVB limiter before applying positivity-preserving limiter
    if (this->all_parameters->limiter_param.use_tvb_limiter == true)
        this->tvbLimiter->limit(solution, dof_handler, fe_collection, volume_quadrature_collection, grid_degree, max_degree, oneD_fe_collection_1state, oneD_quadrature_collection);

    //first limit for positivity of density
    if(pde_type == PDE_enum::euler && nstate == dim + 2){
        this->posdensity_limiter->limit(solution, dof_handler, fe_collection, volume_quadrature_collection, grid_degree, max_degree, oneD_fe_collection_1state, oneD_quadrature_collection);
    }

    //create 1D solution polynomial basis functions and corresponding projection operator
    //to interpolate the solution to the quadrature nodes, and to project it back to the
    //modal coefficients.
    const unsigned int init_grid_degree = grid_degree;

    // Constructor for the operators
    OPERATOR::basis_functions<dim, 2 * dim, real> soln_basis(1, max_degree, init_grid_degree);
    OPERATOR::vol_projection_operator<dim, 2 * dim, real> soln_basis_projection_oper(1, max_degree, init_grid_degree);

    // Build the oneD operator to perform interpolation/projection
    soln_basis.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quadrature_collection[max_degree]);
    soln_basis_projection_oper.build_1D_volume_operator(oneD_fe_collection_1state[max_degree], oneD_quadrature_collection[max_degree]);

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
        const std::vector<real>& quad_weights = volume_quadrature_collection[poly_degree].get_weights();
        std::array<std::vector<real>, nstate> soln_at_q;

        // Interpolate solution dofs to quadrature pts.
        for (int istate = 0; istate < nstate; istate++) {
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }

        // Obtain solution cell average
        std::array<real, nstate> soln_cell_avg = get_soln_cell_avg(soln_at_q, n_quad_pts, quad_weights);

        real entropy_cell_avg = 0.0;
        if(pde_type == PDE_enum::euler && nstate == dim + 2){
            real density_avg = soln_cell_avg[0];
            real pressure_avg = euler_physics->compute_pressure(soln_cell_avg);
           // entropy_cell_avg = euler_physics->compute_entropy(density_avg, pressure_avg);
            entropy_cell_avg = euler_physics->compute_entropy(density_avg, pressure_avg) * density_avg;
        }
        if(pde_type == PDE_enum::burgers_inviscid){
            entropy_cell_avg = 0.5 * soln_cell_avg[0] * soln_cell_avg[0];
        }
        real local_min_entropy_cell = local_min_entropy(cell_index);
       // real entropy_cell_min = -1e9;
        real entropy_cell_min = 1e9;
       // real theta = 1.0;
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
               // theta = std::min(theta, abs((entropy_cell - entropy_cell_avg)/(local_min_entropy_cell - entropy_cell_avg)));
              //  theta = std::min(theta, abs((local_min_entropy_cell - entropy_cell_avg)/(entropy_cell - entropy_cell_avg)));
               // if(entropy_cell>entropy_cell_min)//previous
                if(entropy_cell<entropy_cell_min)
                    entropy_cell_min = entropy_cell;
            }
            if(pde_type == PDE_enum::burgers_inviscid){
                real entropy_cell = 0.5 * soln_state[0] * soln_state[0];
       //         theta = std::min(theta, abs((entropy_cell - entropy_cell_avg)/(local_min_entropy_cell - entropy_cell_avg)));
               // if(entropy_cell>entropy_cell_min)//previous
                if(entropy_cell<entropy_cell_min)
                    entropy_cell_min = entropy_cell;
            }
        }
        real theta = std::min(1.0, abs((local_min_entropy_cell - entropy_cell_avg)/(entropy_cell_min - entropy_cell_avg)));
        //real theta = std::min(1.0, abs((entropy_cell_min - entropy_cell_avg)/(local_min_entropy_cell - entropy_cell_avg)));
       // if(entropy_cell_min > local_min_entropy_cell)
       //     local_min_entropy(cell_index) = entropy_cell_min;

        // Limit values at quadrature points
        for (unsigned int istate = 0; istate < nstate; ++istate) {
            for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
                soln_at_q[istate][iquad] = theta * (soln_at_q[istate][iquad] - soln_cell_avg[istate])
                    + soln_cell_avg[istate];
            }
        }
        //update min
        entropy_cell_min = 1e9;
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
                if(entropy_cell<entropy_cell_min)
                    entropy_cell_min = entropy_cell;
            }
            if(pde_type == PDE_enum::burgers_inviscid){
                real entropy_cell = 0.5 * soln_state[0] * soln_state[0];
                if(entropy_cell<entropy_cell_min)
                    entropy_cell_min = entropy_cell;
            }
        }
        if(entropy_cell_min > local_min_entropy_cell)
            local_min_entropy(cell_index) = entropy_cell_min;

        //this is unecessary bc GLL 
        // Project soln at quadrature points to dofs.
        for (int istate = 0; istate < nstate; istate++) {
            soln_basis_projection_oper.matrix_vector_mult_1D(soln_at_q[istate], soln_coeff[istate], soln_basis_projection_oper.oneD_vol_operator);
        }

        // Write limited solution back and verify that positivity of density is satisfied
        write_limited_solution(solution, soln_coeff, n_shape_fns, current_dofs_indices);
        cell_index++;
    }
}

template class MinEntropyPrincipleLimiter <PHILIP_DIM, 1, double>;
template class MinEntropyPrincipleLimiter <PHILIP_DIM, 2, double>;
template class MinEntropyPrincipleLimiter <PHILIP_DIM, 3, double>;
template class MinEntropyPrincipleLimiter <PHILIP_DIM, 4, double>;
template class MinEntropyPrincipleLimiter <PHILIP_DIM, 5, double>;
template class MinEntropyPrincipleLimiter <PHILIP_DIM, 6, double>;
} // PHiLiP namespace
