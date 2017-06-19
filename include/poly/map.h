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

#include "common/polymorphic_execution.h"
#include "common/support.h"

namespace grppi{

template <typename InputIt, typename OutputIt, typename Operation>
void map_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Operation && op) 
{
}

template <typename InputIt, typename OutputIt, typename Operation,
          typename InputIt2, typename ... OtherInputIts>
void map_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Operation && op, InputIt2 first2, 
         OtherInputIts ... other_its) 
{
}

template <typename E, typename ... O,
          typename InputIt, typename OutputIt, typename Operation,
          internal::requires_execution_not_supported<E> = 0>
void map_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Operation && op) 
{
  map_multi_impl<O...>(e, first, last, first_out, std::forward<Operation>(op));
}

template <typename E, typename ... O,
          typename InputIt, typename OutputIt, typename Operation,
          typename InputIt2, typename ... OtherInputIts,
          internal::requires_execution_not_supported<E> = 0>
void map_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Operation && op, InputIt2 first2, 
         OtherInputIts ... other_its) 
{
  map_multi_impl<O...>(e, first, last, first_out, std::forward<Operation>(op),
                       first2, other_its...);
}


template <typename E, typename ... O,
          typename InputIt, typename OutputIt, typename Operation,
          internal::requires_execution_supported<E> = 0>
void map_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Operation && op) 
{
  if (typeid(E) == e.type()) {
    map(*e.execution_ptr<E>(), 
        first, last, first_out, std::forward<Operation>(op));
  }
  else {
    map_multi_impl<O...>(e, first, last, first_out, std::forward<Operation>(op));
  }
}

template <typename E, typename ... O,
          typename InputIt, typename OutputIt, typename Operation,
          typename InputIt2, typename ... OtherInputIts,
          internal::requires_execution_supported<E> = 0>
void map_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Operation && op, InputIt2 first2, 
         OtherInputIts ... other_its) 
{
  if (typeid(E) == e.type()) {
    map(*e.execution_ptr<E>(), 
        first, last, first_out, std::forward<Operation>(op), first2,
        other_its...);
  }
  else {
    map_multi_impl<O...>(e, first, last, first_out, std::forward<Operation>(op), 
                         first2, other_its...);
  }
}

/// Runs a map pattern with an input sequence, an output sequence and a task
/// function.
/// InputIt: Iterator for the input sequence.
/// OutputIt: Iterator for the output sequence.
/// Operation: Operation functor type
template <typename InputIt, typename OutputIt, typename Operation>
void map(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Operation && op) 
{
  map_multi_impl<
    sequential_execution,
    parallel_execution_thr,
    parallel_execution_omp,
    parallel_execution_tbb
  >(e, first, last, first_out, std::forward<Operation>(op));
}


template <typename InputIt, typename OutputIt, typename InputIt2, typename ... OtherInputIts,
          typename Operation>
void map(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Operation && op, InputIt2 first2, OtherInputIts ... other_its) 
{
  map_multi_impl<
    sequential_execution,
    parallel_execution_thr,
    parallel_execution_omp,
    parallel_execution_tbb
  >(e, first, last, first_out, std::forward<Operation>(op), first2, other_its...);
}


} // end namespace grppi

#endif
