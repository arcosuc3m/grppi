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

#ifndef GRPPI_POLY_STREAM_REDUCE_H
#define GRPPI_POLY_STREAM_REDUCE_H

#include "common/polymorphic_execution.h"
#include "common/support.h"

namespace grppi{

template <typename GenFunc, typename ReduceOperator, typename SinkFunc>
 void stream_reduce_multi_impl(polymorphic_execution & e, GenFunc && in, 
      int windowsize, int offset, ReduceOperator && op, SinkFunc && sink)
{
}

template <typename E, typename ... O,
          typename GenFunc, typename ReduceOperator, typename SinkFunc,
          internal::requires_execution_not_supported<E> = 0>
void stream_reduce_multi_impl(polymorphic_execution & e, GenFunc && in, 
      int windowsize, int offset, ReduceOperator && op, SinkFunc && sink) 
{
  stream_reduce_multi_impl<O...>(e, std::forward<GenFunc>(in),
      windowsize, offset, std::forward<ReduceOperator>(op), 
      std::forward<SinkFunc>(sink));
}



template <typename E, typename ... O,
          typename GenFunc, typename ReduceOperator, typename SinkFunc,
          internal::requires_execution_supported<E> = 0>
void stream_reduce_multi_impl(polymorphic_execution & e, GenFunc && in, 
      int windowsize, int offset, ReduceOperator && op, SinkFunc && sink) 
{
  if (typeid(E) == e.type()) {
    stream_reduce(*e.execution_ptr<E>(), std::forward<GenFunc>(in),
      windowsize, offset, std::forward<ReduceOperator>(op), 
      std::forward<SinkFunc>(sink));
  }
  else {
    stream_reduce_multi_impl<O...>(e, std::forward<GenFunc>(in),
      windowsize, offset, std::forward<ReduceOperator>(op), 
      std::forward<SinkFunc>(sink));
  }
}

/// Runs a stream_reduce pattern with generator function, a reduce function
/// an operation function and output type.
/// GenFunc: Generator functor type.
/// Operation: Operation functor type
/// ReduceFunc: Reductor functor type.
/// OutputType: Output type.
template <typename GenFunc, typename ReduceOperator, typename SinkFunc>
void stream_reduce(polymorphic_execution & e, GenFunc && in, 
      int windowsize, int offset, ReduceOperator && op, SinkFunc && sink) 
{
  stream_reduce_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(e, std::forward<GenFunc>(in),
      windowsize, offset, std::forward<ReduceOperator>(op), 
      std::forward<SinkFunc>(sink));
}


} // end namespace grppi

#endif