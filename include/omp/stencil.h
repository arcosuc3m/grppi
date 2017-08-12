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

#ifndef GRPPI_OMP_STENCIL_H
#define GRPPI_OMP_STENCIL_H

#ifdef GRPPI_OMP

#include "parallel_execution_omp.h"

#include <tuple>
#include <utility>

namespace grppi {

/**
\addtogroup stencil_pattern
@{
\addtogroup stencil_pattern_omp OpenMP Parallel stencil pattern
\brief OpenMP parallel implementation of the \ref md_stencil.
@{
*/

/**
\brief Invoke \ref md_stencil on a data sequence with 
OpenMP parallel execution.
\tparam InputIt Iterator type used for the input sequence.
\tparam OutputIt Iterator type used for the output sequence
\tparam Neighbourhood Callable type for obtaining the neighbourhood.
\tparam StencilTransformer Callable type for performing the stencil transformation.
\param ex OpenMP parallel execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param out Iterator to the first element in the output sequence.
\param transform_op Stencil transformation transform_operation.
\param neighbour_op Neighbourhood transform_operation.
*/
template <typename InputIt, typename OutputIt, typename StencilTransformer, 
          typename Neighbourhood>
void stencil(
    const parallel_execution_omp & ex, 
    InputIt first, InputIt last, OutputIt first_out, 
    StencilTransformer && transform_op, 
    Neighbourhood && neighbour_op) 
{
  ex.stencil(std::make_tuple(first), first_out,
    std::distance(first,last),
    std::forward<StencilTransformer>(transform_op),
    std::forward<Neighbourhood>(neighbour_op));
}

/**
\brief Invoke \ref md_stencil on multiple data sequences with 
TBB parallel execution.
\tparam InputIt Iterator type used for the input sequence.
\tparam OutputIt Iterator type used for the output sequence
\tparam Neighbourhood Callable type for obtaining the neighbourhood.
\tparam StencilTransformer Callable type for performing the stencil transformation.
\tparam OtherInputIts Iterator types for additional input sequences.
\param ex TBB parallel execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param out Iterator to the first element in the output sequence.
\param transform_op Stencil transformation operation.
\param neighbour_op Neighbourhood operation.
\param other_firsts Iterators to the first element of additional input sequences.
*/
template <typename InputIt, typename OutputIt, typename StencilTransformer, 
          typename Neighbourhood, typename ... OtherInputIts>
void stencil(
    const parallel_execution_omp & ex, 
    InputIt first, InputIt last, OutputIt first_out, 
    StencilTransformer && transform_op, 
    Neighbourhood && neighbour_op, 
    OtherInputIts ... other_firsts ) 
{
  ex.stencil(std::make_tuple(first, other_firsts...), first_out,
    std::distance(first,last),
    std::forward<StencilTransformer>(transform_op),
    std::forward<Neighbourhood>(neighbour_op));
}

/**
@}
@}
*/

}

#endif

#endif
