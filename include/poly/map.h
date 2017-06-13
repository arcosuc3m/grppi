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

namespace internal {

template <typename GF, typename TF>
void map_helper(polymorphic_execution & e, GF && generator, TF && task)
{
}

template <typename InputIt, typename OutputIt, typename Operation>
void map_helper(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Operation && op) 
{
}

template <typename E, typename ... O, typename GF, typename TF,
          internal::requires_execution_not_supported<E> = 0>
void map_helper(polymorphic_execution & e, GF && generator, TF && task)
{
  map_helper<O...>(e, std::forward<GF>(generator), std::forward<TF>(task));
}

template <typename E, typename ... O,
          typename InputIt, typename OutputIt, typename Operation,
          internal::requires_execution_not_supported<E> = 0>
void map_helper(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Operation && op) 
{
  map_helper<O...>(e, first, last, first_out, std::forward<Operation>(op));
}

template <typename E, typename ... O, typename GF, typename TF,
          internal::requires_execution_supported<E> = 0>
void map_helper(polymorphic_execution & e, GF && generator, TF && task)
{
  if (typeid(E) == e.type()) {
    map(*e.execution_ptr<E>(), 
        std::forward<GF>(generator), std::forward<TF>(task));
  }
  else {
    map_helper<O...>(std::forward<GF>(generator), std::forward<TF>(task));
  }
}

template <typename E, typename ... O,
          typename InputIt, typename OutputIt, typename Operation,
          internal::requires_execution_supported<E> = 0>
void map_helper(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Operation && op) 
{
  if (typeid(E) == e.type()) {
    map(*e.execution_ptr<E>(), 
        first, last, first_out, std::forward<Operation>(op));
  }
  else {
    map_helper<O...>(e, first, last, first_out, std::forward<Operation>(op));
  }
}

} // end namespace grppi::internal

/// Runs a map pattern with a generator function and a task function.
/// GF: Generator functor type
/// TF: Task functor type
template <typename Generator, typename Operation>
void map(polymorphic_execution & e, Generator && gen, Operation && op)
{
  internal::map_helper<
    sequential_execution,
    parallel_execution_thr,
    parallel_execution_omp,
    parallel_execution_tbb
  >(e, std::forward<Generator>(gen), std::forward<Operation>(op));
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
  internal::map_helper<
    sequential_execution,
    parallel_execution_thr,
    parallel_execution_omp,
    parallel_execution_tbb
  >(e, first, last, first_out, std::forward<Operation>(op));
}

template <typename InputIt, typename OutputIt, typename ... OtherInputIts,
          typename Operation>
void map(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Operation && op, OtherInputIts ... other_its) 
{
  internal::map_helper<
    sequential_execution,
    parallel_execution_thr,
    parallel_execution_omp,
    parallel_execution_tbb
  >(e, first, last, first_out, std::forward<Operation>(op), other_its...);
}

} // end namespace grppi

#endif
