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

#ifndef GRPPI_POLY_MAP_H
#define GRPPI_POLY_MAP_H

#include "polymorphic_execution.h"
#include "../common/support.h"

namespace grppi {

template <typename InputIt, typename OutputIt, typename Transformer>
void map_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Transformer && op) 
{
}

template <typename InputIt, typename OutputIt, typename Transformer,
          typename InputIt2, typename ... OtherInputIts>
void map_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Transformer && op, InputIt2 first2, 
         OtherInputIts ... more_firsts) 
{
}

template <typename E, typename ... O,
          typename InputIt, typename OutputIt, typename Transformer,
          internal::requires_execution_not_supported<E> = 0>
void map_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Transformer && op) 
{
  map_multi_impl<O...>(e, first, last, first_out, std::forward<Transformer>(op));
}

template <typename E, typename ... O,
          typename InputIt, typename OutputIt, typename Transformer,
          typename InputIt2, typename ... OtherInputIts,
          internal::requires_execution_not_supported<E> = 0>
void map_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Transformer && op, InputIt2 first2, 
         OtherInputIts ... more_firsts) 
{
  map_multi_impl<O...>(e, first, last, first_out, std::forward<Transformer>(op),
                       first2, more_firsts...);
}


template <typename E, typename ... O,
          typename InputIt, typename OutputIt, typename Transformer,
          internal::requires_execution_supported<E> = 0>
void map_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Transformer && op) 
{
  if (typeid(E) == e.type()) {
    map(*e.execution_ptr<E>(), 
        first, last, first_out, std::forward<Transformer>(op));
  }
  else {
    map_multi_impl<O...>(e, first, last, first_out, std::forward<Transformer>(op));
  }
}

template <typename E, typename ... O,
          typename InputIt, typename OutputIt, typename Transformer,
          typename InputIt2, typename ... OtherInputIts,
          internal::requires_execution_supported<E> = 0>
void map_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Transformer && op, InputIt2 first2, 
         OtherInputIts ... more_firsts) 
{
  if (typeid(E) == e.type()) {
    map(*e.execution_ptr<E>(), 
        first, last, first_out, std::forward<Transformer>(op), first2,
        more_firsts...);
  }
  else {
    map_multi_impl<O...>(e, first, last, first_out, std::forward<Transformer>(op), 
                         first2, more_firsts...);
  }
}

/**
\addtogroup map_pattern
@{
\addtogroup map_pattern_poly Polymorphic execution map pattern.
\brief Polymorphic implementation of \ref md_map.
@{
*/

/**
\brief Invoke \ref map-pattern on a data sequence with
polymorphic execution.
\tparam InputIt Iterator type used for input sequence.
\tparam OtuputIt Iterator type used for the output sequence.
\tparam Transformer Callable type for the transformation operation.
\param ex Polymorphic execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param first_out Iterator to first elemento of the output sequence.
\param transf_op Transformation operation.
*/
template <typename InputIt, typename OutputIt, typename Transformer>
void map(polymorphic_execution & ex, 
         InputIt first, InputIt last, OutputIt first_out, 
         Transformer && transf_op) 
{
  map_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(ex, first, last, first_out, std::forward<Transformer>(transf_op));
}

/**
\brief Invoke \ref map-pattern on a data sequence with
polymorphic parallel execution.
\tparam InputIt Iterator type used for input sequence.
\tparam OtuputIt Iterator type used for the output sequence.
\tparam Transformer Callable type for the transformation operation.
\tparam OtherInputIts Iterator types used for additional input sequences.
\param ex Polymorphic execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param first_out Iterator to first elemento of the output sequence.
\param op Transformation operation.
\param more_firsts Additional iterators with first elements of additional sequences.
*/
template <typename InputIt, typename OutputIt, typename InputIt2, 
          typename Transformer,
          typename ... OtherInputIts>
void map(polymorphic_execution & ex, InputIt first, InputIt last, 
         OutputIt first_out, Transformer && op, InputIt2 first2, OtherInputIts ... more_firsts) 
{
  map_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(ex, first, last, first_out, std::forward<Transformer>(op), first2, more_firsts...);
}

/**
@}
@}
*/

} // end namespace grppi

#endif
