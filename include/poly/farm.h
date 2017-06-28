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

#ifndef GRPPI_POLY_FARM_H
#define GRPPI_POLY_FARM_H

#include "common/polymorphic_execution.h"
#include "common/support.h"

namespace grppi{

template <typename GF, typename Operation>
void farm_multi_impl(polymorphic_execution & e, GF &&in, Operation && op) 
{
}


template <typename GF, typename Operation, typename SinkF>
void farm_multi_impl(polymorphic_execution & e, GF && in, Operation && op, SinkF &&sink) 
{
}


template <typename E, typename ... O, typename GF, typename Operation,
          internal::requires_execution_not_supported<E> = 0>
void farm_multi_impl(polymorphic_execution & e, GF && in, Operation && op)
{
  farm_multi_impl<O...>(e, std::forward<GF>(in), std::forward<Operation>(op));
}

template <typename E, typename ... O, typename GF, typename Operation, typename SinkF,
          internal::requires_execution_not_supported<E> = 0>
void farm_multi_impl(polymorphic_execution & e, GF && in, Operation && op, SinkF && sink)
{
  farm_multi_impl<O...>(e, std::forward<GF>(in), std::forward<Operation>(op), std::forward<SinkF>(sink));
}



template <typename E, typename ... O, typename GF, typename Operation,
          internal::requires_execution_supported<E> = 0>
void farm_multi_impl(polymorphic_execution & e, GF && in, Operation && op)
{
  if (typeid(E) == e.type()) {
    farm(*e.execution_ptr<E>(), 
        std::forward<GF>(in), std::forward<Operation>(op));
  }
  else {
    farm_multi_impl<O...>(e, std::forward<GF>(in), std::forward<Operation>(op));
  }
}


template <typename E, typename ... O, typename GF, typename Operation, typename SinkF,
          internal::requires_execution_supported<E> = 0>
void farm_multi_impl(polymorphic_execution & e, GF && in, Operation && op, SinkF && sink)
{
  if (typeid(E) == e.type()) {
    farm(*e.execution_ptr<E>(), 
        std::forward<GF>(in), std::forward<Operation>(op), std::forward<SinkF>(sink));
  }
  else {
    farm_multi_impl<O...>(e, std::forward<GF>(in), std::forward<Operation>(op), std::forward<SinkF>(sink));
  }
}



/// Runs a farm pattern with a generator function and an operation function
/// GF: Generator functor type.
/// Operation: Operation functor type.
/// SinkF: Sink functor type.
template <typename GF, typename Operation>
void farm(polymorphic_execution & e, GF && in, Operation && op)
{
  farm_multi_impl<
    sequential_execution,
    parallel_execution_thr,
    parallel_execution_omp,
    parallel_execution_tbb
  >(e, std::forward<GF>(in), std::forward<Operation>(op));
}

template <typename GF, typename Operation, typename SinkF>
void farm(polymorphic_execution & e, GF && in, Operation && op, SinkF && sink)
{
  farm_multi_impl<
    sequential_execution,
    parallel_execution_thr,
    parallel_execution_omp,
    parallel_execution_tbb
  >(e, std::forward<GF>(in), std::forward<Operation>(op), std::forward<SinkF>(sink));
}




} // end namespace grppi

#endif