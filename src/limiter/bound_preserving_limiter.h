#ifndef __BOUND_PRESERVING_LIMITER__
#define __BOUND_PRESERVING_LIMITER__

#include "physics/physics.h"
#include "parameters/all_parameters.h"
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/base/qprojector.h>
#include <deal.II/base/geometry_info.h>

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q1_eulerian.h>


#include <deal.II/dofs/dof_handler.h>

#include <deal.II/hp/q_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/fe_values.h>

#include "dg/dg.h"

namespace PHiLiP {
        template<int dim, typename real>
        class BoundPreservingLimiter
        {
        public:
            /// Constructor
            BoundPreservingLimiter(
                const int nstate_input,//number of states input
                const Parameters::AllParameters* const parameters_input);//pointer to parameters

            /// Destructor
            ~BoundPreservingLimiter() {};

            ///Number of states
            const int nstate;

            /// Pointer to parameters object
            const Parameters::AllParameters* const all_parameters;

            virtual void limit(
                dealii::LinearAlgebra::distributed::Vector<double>& solution,
                const dealii::DoFHandler<dim>& dof_handler,
                const dealii::hp::FECollection<dim>& fe_collection,
                dealii::hp::QCollection<dim>                            volume_quadrature_collection,
                unsigned int                                            tensor_degree,
                unsigned int                                            max_degree,
                const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
                dealii::hp::QCollection<1>                              oneD_quadrature_collection) = 0;

        }; // End of BoundPreservingLimiter Class

        template<int dim, int nstate, typename real>
        class TVBLimiter : public BoundPreservingLimiter <dim, real>
        {
        public:
            /// Constructor
            TVBLimiter(
                const Parameters::AllParameters* const parameters_input);//max poly degree

            /// Destructor
            ~TVBLimiter() {};

            void limit(
                dealii::LinearAlgebra::distributed::Vector<double>& solution,
                const dealii::DoFHandler<dim>& dof_handler,
                const dealii::hp::FECollection<dim>& fe_collection,
                dealii::hp::QCollection<dim>                            volume_quadrature_collection,
                unsigned int                                            tensor_degree,
                unsigned int                                            max_degree,
                const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
                dealii::hp::QCollection<1>                              oneD_quadrature_collection);

        }; // End of TVBLimiter Class

        template<int dim, int nstate, typename real>
        class MaximumPrincipleLimiter : public BoundPreservingLimiter <dim, real>
        {
        public:
            /// Constructor
            MaximumPrincipleLimiter(
                const Parameters::AllParameters* const parameters_input);

            /// Destructor
            ~MaximumPrincipleLimiter() {};

            /// Initial global maximum of solution in domain.
            std::vector<real> global_max;
            /// Initial global minimum of solution in domain.
            std::vector<real> global_min;

            void get_global_max_and_min_of_solution(
                dealii::LinearAlgebra::distributed::Vector<double>      solution,
                const dealii::DoFHandler<dim>&                          dof_handler,
                const dealii::hp::FECollection<dim>&                    fe_collection);

            void limit(
                dealii::LinearAlgebra::distributed::Vector<double>& solution,
                const dealii::DoFHandler<dim>& dof_handler,
                const dealii::hp::FECollection<dim>& fe_collection,
                dealii::hp::QCollection<dim>                            volume_quadrature_collection,
                unsigned int                                            tensor_degree,
                unsigned int                                            max_degree,
                const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
                dealii::hp::QCollection<1>                              oneD_quadrature_collection);

        }; // End of MaximumPrincipleLimiter Class

        template<int dim, int nstate, typename real>
        class PositivityPreservingLimiter : public BoundPreservingLimiter <dim, real>
        {
        public:
            /// Constructor
            PositivityPreservingLimiter(
                const Parameters::AllParameters* const parameters_input);

            /// Destructor
            ~PositivityPreservingLimiter() {};

            void limit(
                dealii::LinearAlgebra::distributed::Vector<double>& solution,
                const dealii::DoFHandler<dim>& dof_handler,
                const dealii::hp::FECollection<dim>& fe_collection,
                dealii::hp::QCollection<dim>                            volume_quadrature_collection,
                unsigned int                                            tensor_degree,
                unsigned int                                            max_degree,
                const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
                dealii::hp::QCollection<1>                              oneD_quadrature_collection);

        }; // End of PositivityPreservingLimiter Class

        template<int dim, int nstate, typename real>
        class PositivityPreservingLimiterRobust : public BoundPreservingLimiter <dim, real>
        {
        public:
            /// Constructor
            PositivityPreservingLimiterRobust(
                const Parameters::AllParameters* const parameters_input);

            /// Destructor
            ~PositivityPreservingLimiterRobust() {};

            void limit(
                dealii::LinearAlgebra::distributed::Vector<double>& solution,
                const dealii::DoFHandler<dim>& dof_handler,
                const dealii::hp::FECollection<dim>& fe_collection,
                dealii::hp::QCollection<dim>                            volume_quadrature_collection,
                unsigned int                                            tensor_degree,
                unsigned int                                            max_degree,
                const dealii::hp::FECollection<1>                       oneD_fe_collection_1state,
                dealii::hp::QCollection<1>                              oneD_quadrature_collection);

        }; // End of PositivityPreservingLimiterRobust Class
} // PHiLiP namespace

#endif
