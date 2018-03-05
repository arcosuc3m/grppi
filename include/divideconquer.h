/**
* @version		GrPPI v0.3
* @copyright		Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license		GNU/GPL, see LICENSE.txt
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
#ifndef GRPPI_DIVIDECONQUER_H
#define GRPPI_DIVIDECONQUER_H

#include <utility>

#include "common/execution_traits.h"

namespace grppi {

/** 
\addtogroup task_patterns
@{
\defgroup divide_conquer_pattern Divide/Conquer pattern
\brief Interface for applyinng the \ref md_divide-conquer.
@{
*/

/**
\brief Invoke \ref md_divide-conquer.
\parapm Execution Execution type.
\tparam Input Type used for the input problem.
\tparam Divider Callable type for the divider operation.
\tparam Solver Callable type for the solver operation.
\tparam Combiner Callable type for the combiner operation.
\param ex Execution policy object.
\param input Input problem to be solved.
\param divider_op Divider operation.
\param solver_op Solver operation.
\param combiner_op Combiner operation.
*/
template <typename Execution, typename Input, 
          typename Divider, typename Solver, typename Combiner>
[[deprecated("Use newer divide_conquer with predicate arguemnt")]]
auto divide_conquer(
    const Execution & ex, 
    Input && input, 
    Divider && divider_op, 
    Solver && solver_op, 
    Combiner && combiner_op) 
{
  static_assert(supports_divide_conquer<Execution>(),
      "divide/conquer pattern not supported for execution type");
  return ex.divide_conquer(std::forward<Input>(input), 
        std::forward<Divider>(divider_op), std::forward<Solver>(solver_op), 
        std::forward<Combiner>(combiner_op));
}

/**
\brief Invoke \ref md_divide-conquer.
\parapm Execution Execution type.
\tparam Input Type used for the input problem.
\tparam Divider Callable type for the divider operation.
\tparam Predicate Callable type for the stop condition predicate.
\tparam Solver Callable type for the solver operation.
\tparam Combiner Callable type for the combiner operation.
\param ex Execution policy object.
\param input Input problem to be solved.
\param divider_op Divider operation.
\param predicate_op Predicate operation.
\param solver_op Solver operation.
\param combiner_op Combiner operation.
*/
template <typename Execution, typename Input,
          typename Divider,typename Predicate, typename Solver, typename Combiner>
auto divide_conquer(
    const Execution & ex,
    Input && input,
    Divider && divider_op,
    Predicate && predicate_op,
    Solver && solver_op,
    Combiner && combiner_op)
{
  static_assert(supports_divide_conquer<Execution>(),
      "divide/conquer pattern not supported for execution type");
  return ex.divide_conquer(std::forward<Input>(input),
        std::forward<Divider>(divider_op),
        std::forward<Predicate>(predicate_op),
        std::forward<Solver>(solver_op),
        std::forward<Combiner>(combiner_op));
}

/**
@}
@}
*/

}

#endif
