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

#ifndef GRPPI_POLY_STREAM_ITERATOR_H
#define GRPPI_POLY_STREAM_ITERATOR_H

#include "common/support.h"
#include "polymorphic_execution.h"

namespace grppi{

/*template<typename GenFunc, typename Predicate, typename OutFunc, typename ...Stages>
 void stream_iteration(parallel_execution_native &p, GenFunc && in, pipeline_info<parallel_execution_native , Stages...> && se, Predicate && condition, OutFunc && out){

-->
(CREO un typename nuevo y se lo paso)
template<typename GenFunc, typename Predicate, typename OutFunc,
         typename Object, typename ...Stages>
 void stream_iteration(parallel_execution_native &p, GenFunc && in, 
      Object && se, Predicate && condition, OutFunc && out){*/



template<typename GenFunc, typename Predicate, typename OutFunc,
         typename Operation>
void stream_iteration_multi_impl(polymorphic_execution & e, GenFunc && in, 
      Operation && op, Predicate && condition, OutFunc && out)
{
}



template <typename E, typename ... O,
          typename GenFunc, typename Predicate, typename OutFunc,
          typename Operation,
          internal::requires_execution_not_supported<E> = 0>
void stream_iteration_multi_impl(polymorphic_execution & e,  GenFunc && in, 
      Operation && op, Predicate && condition, OutFunc && out) 
{
  stream_iteration_multi_impl<O...>(e, std::forward<GenFunc>(in), 
    std::forward<Operation>(op), std::forward<Predicate>(condition), 
    std::forward<OutFunc>(out));
}




template <typename E, typename ... O,
          typename GenFunc, typename Predicate, typename OutFunc,
          typename Operation,
          internal::requires_execution_supported<E> = 0>
void stream_iteration_multi_impl(polymorphic_execution & e, GenFunc && in, 
      Operation && op, Predicate && condition, OutFunc && out) 
{
  if (typeid(E) == e.type()) {
    stream_iteration(*e.execution_ptr<E>(), 
        std::forward<GenFunc>(in), 
        std::forward<Operation>(op), std::forward<Predicate>(condition), 
        std::forward<OutFunc>(out));
  }
  else {
    stream_iteration_multi_impl<O...>(e, std::forward<GenFunc>(in), 
        std::forward<Operation>(op), std::forward<Predicate>(condition), 
        std::forward<OutFunc>(out));
  }
}



/// Runs a stream_iteration pattern with a generator function, an operation object
/// a predicate condition and an output function.
/// GenFunc: Generator functor type.
/// Operation: Operation functor type.
/// Predicate: Predicator functor type.
/// OutFunc: Output functor type.
template <typename GenFunc, typename Predicate, typename OutFunc,
          typename Operation>
void stream_iteration(polymorphic_execution & e, GenFunc && in, 
      Operation && op, Predicate && condition, OutFunc && out) 
{
  stream_iteration_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(e, std::forward<GenFunc>(in), std::forward<Operation>(op), 
       std::forward<Predicate>(condition), std::forward<OutFunc>(out));
}



} // end namespace grppi

#endif
