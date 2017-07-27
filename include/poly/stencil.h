/**
* @version		GrPPI v0.2
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

#ifndef GRPPI_POLY_STENCIL_H
#define GRPPI_POLY_STENCIL_H

#include "polymorphic_execution.h"
#include "../common/support.h"

namespace grppi {

template <typename InputIt, typename OutputIt, typename StencilTransformer, typename Neighbourhood>
void stencil_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, StencilTransformer && op, Neighbourhood && neighbor) 
{
}

template <typename InputIt, typename OutputIt, typename StencilTransformer,
          typename Neighbourhood, typename ... OtherInputIts>
void stencil_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, StencilTransformer && op, Neighbourhood && neighbor, 
         OtherInputIts ... other_its) 
{
}



template <typename E, typename ... O,
          typename InputIt, typename OutputIt, typename StencilTransformer, typename Neighbourhood,
          internal::requires_execution_not_supported<E> = 0>
void stencil_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, StencilTransformer && op, Neighbourhood && neighbor) 
{
  stencil_multi_impl<O...>(e, first, last, first_out, std::forward<StencilTransformer>(op),
         std::forward<Neighbourhood>(neighbor));
}

template <typename E, typename ... O,
          typename InputIt, typename OutputIt, typename StencilTransformer,
          typename Neighbourhood, typename ... OtherInputIts,
          internal::requires_execution_not_supported<E> = 0>
void stencil_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, StencilTransformer && op, Neighbourhood && neighbor, 
         OtherInputIts ... other_its) 
{
  stencil_multi_impl<O...>(e, first, last, first_out, std::forward<StencilTransformer>(op),
         std::forward<Neighbourhood>(neighbor), other_its...);
}



template <typename E, typename ... O,
          typename InputIt, typename OutputIt, typename StencilTransformer, typename Neighbourhood,
          internal::requires_execution_supported<E> = 0>
void stencil_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, StencilTransformer && op, Neighbourhood && neighbor) 
{
  if (typeid(E) == e.type()) {
    stencil(*e.execution_ptr<E>(), 
        first, last, first_out, std::forward<StencilTransformer>(op),
        std::forward<Neighbourhood>(neighbor));
  }
  else {
    stencil_multi_impl<O...>(e, first, last, first_out, std::forward<StencilTransformer>(op),
        std::forward<Neighbourhood>(neighbor));
  }
}

template <typename E, typename ... O,
          typename InputIt, typename OutputIt, typename StencilTransformer,
          typename Neighbourhood, typename ... OtherInputIts,
          internal::requires_execution_supported<E> = 0>
void stencil_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, StencilTransformer && op, Neighbourhood && neighbor, 
         OtherInputIts ... other_its) 
{
  if (typeid(E) == e.type()) {
    stencil(*e.execution_ptr<E>(), 
        first, last, first_out, std::forward<StencilTransformer>(op), std::forward<Neighbourhood>(neighbor),
        other_its...);
  }
  else {
    stencil_multi_impl<O...>(e, first, last, first_out, std::forward<StencilTransformer>(op), 
        std::forward<Neighbourhood>(neighbor), other_its...);
  }
}

/**
\addtogroup stencil_pattern
@{
\addtogroup stencil_pattern_poly Polymorphic stencil pattern
\brief Polymorphic implementation of the \ref md_stencil.
@{
*/

/**
\brief Invoke \ref md_stencil on a data sequence with 
polymorphic execution.
\tparam InputIt Iterator type used for the input sequence.
\tparam OutputIt Iterator type used for the output sequence
\tparam Neighbourhood Callable type for obtaining the neighbourhood.
\tparam StencilTransformer Callable type for performing the stencil transformation.
\param ex Polymorphic execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param out Iterator to the first element in the output sequence.
\param transform_op Stencil transformation operation.
\param neighbour_op Neighbourhood operation.
*/
template <typename InputIt, typename OutputIt, typename StencilTransformer, 
          typename Neighbourhood>
void stencil(polymorphic_execution & ex, 
             InputIt first, InputIt last, OutputIt first_out, 
             StencilTransformer && transform_op, Neighbourhood && neighbour_op) 
{
  stencil_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(ex, first, last, first_out, 
      std::forward<StencilTransformer>(transform_op),
      std::forward<Neighbourhood>(neighbour_op));
}

/**
\brief Invoke \ref md_stencil on multiple data sequences with 
sequential execution.
\tparam InputIt Iterator type used for the input sequence.
\tparam OutputIt Iterator type used for the output sequence
\tparam Neighbourhood Callable type for obtaining the neighbourhood.
\tparam StencilTransformer Callable type for performing the stencil transformation.
\tparam OtherInputIts Iterator types for additional input sequences.
\param ex Sequential execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param out Iterator to the first element in the output sequence.
\param transform_op Stencil transformation operation.
\param neighbour_op Neighbourhood operation.
\param other_firsts Iterators to the first element of additional input sequences.
*/
template <typename InputIt, typename OutputIt, typename StencilTransformer,
          typename Neighbourhood, typename ... OtherInputIts>
void stencil(polymorphic_execution & ex, 
             InputIt first, InputIt last, OutputIt first_out, 
             StencilTransformer && transform_op, Neighbourhood && neighbour_op, 
             OtherInputIts ... other_its) 
{
  stencil_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(ex, first, last, first_out, 
      std::forward<StencilTransformer>(transform_op), 
      std::forward<Neighbourhood>(neighbour_op), 
      other_its...);
}

/**
@}
@}
*/

} // end namespace grppi

#endif
