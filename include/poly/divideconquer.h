/**
* @version    GrPPI v0.2
* @copyright    Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license    GNU/GPL, see LICENSE.txt
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You have received a copy of the GNU General Public License in LICENSE.txt
* also available in <http://www.gnu.org/licenses/gpl.html>.
*
* See COPYRIGHT.txt for copyright notices and details.
*/

#ifndef GRPPI_POLY_DIVIDECONQUER_H
#define GRPPI_POLY_DIVIDECONQUER_H

#include "polymorphic_execution.h"
#include "common/support.h"

namespace grppi{

template <typename Input, typename Divider, typename Solver, typename Combiner>
typename std::result_of<Solver(Input)>::type 
divide_conquer_multi_impl(polymorphic_execution & ex, 
                              Input & input, 
                              Divider && divide_op, Solver && solve_op, 
                              Combiner && combine_op)
{
  return {};
}

template <typename E, typename ... O,
          typename Input, typename Divider, typename Solver, typename Combiner,
          internal::requires_execution_not_supported<E> = 0>
typename std::result_of<Solver(Input)>::type 
divide_conquer_multi_impl(polymorphic_execution & ex, 
                              Input & input, 
                              Divider && divide_op, Solver && solve_op, 
                              Combiner && combine_op) 
{
  return divide_conquer_multi_impl<O...>(ex, input,  
      std::forward<Divider>(divide_op), std::forward<Solver>(solve_op),
      std::forward<Combiner>(combine_op) );
}

template <typename E, typename ... O,
          typename Input, typename Divider, typename Solver, typename Combiner,
          internal::requires_execution_supported<E> = 0>
typename std::result_of<Solver(Input)>::type 
divide_conquer_multi_impl(polymorphic_execution & ex, 
                              Input & input, 
                              Divider && divide_op, Solver && solve_op, 
                              Combiner && combine_op) 
{
  if (typeid(E) == ex.type()) {
    return divide_conquer(*ex.execution_ptr<E>(), 
        input, 
        std::forward<Divider>(divide_op), std::forward<Solver>(solve_op),
        std::forward<Combiner>(combine_op));
  }
  else {
    return divide_conquer_multi_impl<O...>(ex, input,  
        std::forward<Divider>(divide_op), std::forward<Solver>(solve_op),
        std::forward<Combiner>(combine_op));
  }
}

/**
\addtogroup divide_conquer_pattern
@{
*/

/**
\addtogroup divide_conquer_pattern_poly Polymorphic execution divide/conquer
pattern.
\brief Polymorphic execution implementation of the \ref md_divide-conquer pattern.
@{
*/

/**
\brief Invoke [divide/conquer pattern](@ref md_divide-conquer) with polymorphic
execution.
\tparam Input Type used for the input input.
\tparam Divider Callable type for the divider operation.
\tparam Solver Callable type for the solver operation.
\tparam Combiner Callable type for the combiner operation.
\param ex Sequential execution policy object.
\param input Input input to be solved.
\param divider_op Divider operation.
\param solver_op Solver operation.
\param combiner_op Combiner operation.
*/
template <typename Input, typename Divider, typename Solver, typename Combiner>
typename std::result_of<Solver(Input)>::type 
divide_conquer(polymorphic_execution & ex, 
                   Input & input, 
                   Divider && divide_op, Solver && solve_op, 
                   Combiner && combine_op) 
{
  return divide_conquer_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(ex, input,  
      std::forward<Divider>(divide_op), std::forward<Solver>(solve_op),
      std::forward<Combiner>(combine_op));
}

/**
@}
@
*/

} // end namespace grppi

#endif
