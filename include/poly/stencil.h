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

#include "common/polymorphic_execution.h"
#include "common/support.h"

namespace grppi{

template <typename InputIt, typename OutputIt, typename Operation, typename NFunc>
void stencil_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Operation && op, NFunc && neighbor) 
{
}

template <typename InputIt, typename OutputIt, typename Operation,
          typename NFunc, typename ... OtherInputIts>
void stencil_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Operation && op, NFunc && neighbor, 
         OtherInputIts ... other_its) 
{
}



template <typename E, typename ... O,
          typename InputIt, typename OutputIt, typename Operation, typename NFunc,
          internal::requires_execution_not_supported<E> = 0>
void stencil_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Operation && op, NFunc && neighbor) 
{
  stencil_multi_impl<O...>(e, first, last, first_out, std::forward<Operation>(op),
         std::forward<NFunc>(neighbor));
}

template <typename E, typename ... O,
          typename InputIt, typename OutputIt, typename Operation,
          typename NFunc, typename ... OtherInputIts,
          internal::requires_execution_not_supported<E> = 0>
void stencil_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Operation && op, NFunc && neighbor, 
         OtherInputIts ... other_its) 
{
  stencil_multi_impl<O...>(e, first, last, first_out, std::forward<Operation>(op),
         std::forward<NFunc>(neighbor), other_its...);
}



template <typename E, typename ... O,
          typename InputIt, typename OutputIt, typename Operation, typename NFunc,
          internal::requires_execution_supported<E> = 0>
void stencil_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Operation && op, NFunc && neighbor) 
{
  if (typeid(E) == e.type()) {
    stencil(*e.execution_ptr<E>(), 
        first, last, first_out, std::forward<Operation>(op),
        std::forward<NFunc>(neighbor));
  }
  else {
    stencil_multi_impl<O...>(e, first, last, first_out, std::forward<Operation>(op),
        std::forward<NFunc>(neighbor));
  }
}

template <typename E, typename ... O,
          typename InputIt, typename OutputIt, typename Operation,
          typename NFunc, typename ... OtherInputIts,
          internal::requires_execution_supported<E> = 0>
void stencil_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Operation && op, NFunc && neighbor, 
         OtherInputIts ... other_its) 
{
  if (typeid(E) == e.type()) {
    stencil(*e.execution_ptr<E>(), 
        first, last, first_out, std::forward<Operation>(op), std::forward<NFunc>(neighbor),
        other_its...);
  }
  else {
    stencil_multi_impl<O...>(e, first, last, first_out, std::forward<Operation>(op), 
        std::forward<NFunc>(neighbor), other_its...);
  }
}

/// Runs a stencil pattern with an input sequence, an output sequence
/// an operation function and a neighbor function.
/// InputIt: Iterator for the input sequence.
/// OutputIt: Iterator for the output sequence.
/// Operation: Operation functor type.
/// NFunc: Neighbor functor type.
template <typename InputIt, typename OutputIt, typename Operation, typename NFunc>
void stencil(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Operation && op, NFunc && neighbor) 
{
  stencil_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(e, first, last, first_out, std::forward<Operation>(op),
      std::forward<NFunc>(neighbor));
}


template <typename InputIt, typename OutputIt, typename Operation,
          typename NFunc, typename ... OtherInputIts>
void stencil(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, Operation && op, NFunc && neighbor, 
         OtherInputIts ... other_its) 
{
  stencil_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(e, first, last, first_out, std::forward<Operation>(op), 
      std::forward<NFunc>(neighbor), other_its...);
}


} // end namespace grppi

#endif
