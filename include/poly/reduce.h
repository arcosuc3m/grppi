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

#ifndef GRPPI_POLY_REDUCE_H
#define GRPPI_POLY_REDUCE_H

#include "common/polymorphic_execution.h"
#include "common/support.h"

namespace grppi{

template <typename InputIt, typename Output, typename ReduceOperator>
typename std::enable_if<!is_iterator<Output>::value, void>::type
 reduce_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         Output & out, ReduceOperator op) 
{
}

template <typename InputIt, typename ReduceOperator>
typename ReduceOperator::result_type
 reduce_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
        ReduceOperator op) 
{
}

template <typename InputIt, typename OutputIt, typename RedFunc>
typename  std::enable_if<is_iterator<OutputIt>::value, void>::type
 reduce_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, RedFunc && op) 
{
}



template <typename E, typename ... O,
          typename InputIt, typename Output, typename ReduceOperator,
          internal::requires_execution_not_supported<E> = 0>
typename std::enable_if<!is_iterator<Output>::value, void>::type
 reduce_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         Output & out, ReduceOperator op) 
{
  reduce_multi_impl<O...>(e, first, last, out, op);
}

template <typename E, typename ... O,
          typename InputIt, typename ReduceOperator,
          internal::requires_execution_not_supported<E> = 0>
typename ReduceOperator::result_type
 reduce_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         ReduceOperator op) 
{
  reduce_multi_impl<O...>(e, first, last, op);
}

template <typename E, typename ... O,
          typename InputIt, typename OutputIt, typename RedFunc,
          internal::requires_execution_not_supported<E> = 0>
typename  std::enable_if<is_iterator<OutputIt>::value, void>::type
 reduce_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, RedFunc && reduce) 
{
  reduce_multi_impl<O...>(e, first, last, first_out, std::forward<RedFunc>(reduce));
}



template <typename E, typename ... O,
          typename InputIt, typename Output, typename ReduceOperator,
          internal::requires_execution_supported<E> = 0>
typename std::enable_if<!is_iterator<Output>::value, void>::type
 reduce_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         Output & out, ReduceOperator op) 
{
  if (typeid(E) == e.type()) {
    reduce(*e.execution_ptr<E>(), 
        first, last, out, op);
  }
  else {
    reduce_multi_impl<O...>(e, first, last, out, op);
  }
}

template <typename E, typename ... O,
          typename InputIt, typename ReduceOperator,
          internal::requires_execution_supported<E> = 0>
typename ReduceOperator::result_type
 reduce_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         ReduceOperator op) 
{
  if (typeid(E) == e.type()) {
    reduce(*e.execution_ptr<E>(), 
        first, last, op);
  }
  else {
    reduce_multi_impl<O...>(e, first, last, op);
  }
}

template <typename E, typename ... O,
          typename InputIt, typename OutputIt, typename RedFunc,
          internal::requires_execution_supported<E> = 0>
typename  std::enable_if<is_iterator<OutputIt>::value, void>::type
 reduce_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, RedFunc && reduce) 
{
  if (typeid(E) == e.type()) {
    reduce(*e.execution_ptr<E>(), 
        first, last, first_out, std::forward<RedFunc>(reduce));
  }
  else {
    reduce_multi_impl<O...>(e, first, last, first_out, std::forward<RedFunc>(reduce));
  }
}


/// Runs a reduce pattern with an input sequence, an output sequence and a 
/// reduce operation.
/// InputIt: Iterator for the input sequence.
/// OutputIt: Iterator for the output sequence.
/// ReduceOperator: Reduction functor type
template <typename InputIt, typename Output, typename ReduceOperator>
typename std::enable_if<!is_iterator<Output>::value, void>::type
 reduce(polymorphic_execution & e, InputIt first, InputIt last, 
         Output & out, ReduceOperator op) 
{
  reduce_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(e, first, last, out, op);
}

template <typename InputIt, typename ReduceOperator>
typename ReduceOperator::result_type
 reduce(polymorphic_execution & e, InputIt first, InputIt last, 
         ReduceOperator op) 
{
  reduce_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(e, first, last, op);
}

template <typename InputIt, typename OutputIt, typename RedFunc>
typename  std::enable_if<is_iterator<OutputIt>::value, void>::type
 reduce(polymorphic_execution & e, InputIt first, InputIt last, 
         OutputIt first_out, RedFunc && reduce) 
{
  reduce_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(e, first, last, first_out, std::forward<RedFunc>(reduce));
}


} // end namespace grppi

#endif
