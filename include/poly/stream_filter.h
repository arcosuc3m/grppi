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

#ifndef GRPPI_POLY_STREAM_FILTER_H
#define GRPPI_POLY_STREAM_FILTER_H

#include "common/support.h"
#include "polymorphic_execution.h"

namespace grppi{

template <typename GenFunc, typename FilterFunc, typename OutFunc>
void stream_filter_multi_impl(polymorphic_execution & e, GenFunc && in,  
         FilterFunc && filter, OutFunc && out) 
{
}



template <typename E, typename ... O,
          typename GenFunc, typename FilterFunc, typename OutFunc,
          internal::requires_execution_not_supported<E> = 0>
void stream_filter_multi_impl(polymorphic_execution & e, GenFunc && in, 
         FilterFunc && filter, OutFunc && out) 
{
  stream_filter_multi_impl<O...>(e, std::forward<GenFunc>(in),
    std::forward<FilterFunc>(filter), std::forward<OutFunc>(out));
}



template <typename E, typename ... O,
          typename GenFunc, typename FilterFunc, typename OutFunc,
          internal::requires_execution_supported<E> = 0>
void stream_filter_multi_impl(polymorphic_execution & e, GenFunc && in, 
         FilterFunc && filter, OutFunc && out) 
{
  if (typeid(E) == e.type()) {
    stream_filter(*e.execution_ptr<E>(), 
      std::forward<GenFunc>(in), std::forward<FilterFunc>(filter), 
      std::forward<OutFunc>(out));
  }
  else {
    stream_filter_multi_impl<O...>(e, std::forward<GenFunc>(in), std::forward<FilterFunc>(filter),
      std::forward<OutFunc>(out));
  }
}



/// Runs a stream_filter pattern with a generator function, a filter function
/// and a output function.
/// GenFunc: Generator functor type.
/// FilterFunc: Filter functor type.
/// OutFunc: Output functor type.
template <typename GenFunc, typename FilterFunc, typename OutFunc>
void stream_filter(polymorphic_execution & e, GenFunc && in, 
         FilterFunc && filter, OutFunc && out) 
{
  stream_filter_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(e, std::forward<GenFunc>(in), std::forward<FilterFunc>(filter),
      std::forward<OutFunc>(out));
}



} // end namespace grppi

#endif
