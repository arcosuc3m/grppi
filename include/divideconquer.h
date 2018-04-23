/*
 * Copyright 2018 Universidad Carlos III de Madrid
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
