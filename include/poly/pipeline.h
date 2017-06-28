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

#ifndef GRPPI_POLY_PIPELINE_H
#define GRPPI_POLY_PIPELINE_H

#include "common/polymorphic_execution.h"
#include "common/support.h"

namespace grppi{
template <typename FuncIn, typename ...Stages>
void pipeline_multi_impl(polymorphic_execution & e, FuncIn && in, Stages && ... sgs) 
{
}


template <typename E, typename ... O,
          typename FuncIn, typename ...Stages,
          internal::requires_execution_not_supported<E> = 0>
void pipeline_multi_impl(polymorphic_execution & e, FuncIn && in, Stages && ... sgs) 
{
  pipeline_multi_impl<O...>(e, std::forward<FuncIn>(in), 
    std::forward<Stages>(sgs) ...);
}




template <typename E, typename ... O,
          typename FuncIn, typename ...Stages,
          internal::requires_execution_supported<E> = 0>
void pipeline_multi_impl(polymorphic_execution & e, FuncIn && in, Stages && ... sgs) 
{
  if (typeid(E) == e.type()) {
    pipeline(*e.execution_ptr<E>(), 
      std::forward<FuncIn>(in), std::forward<Stages>(sgs) ...);
  }
  else {
    pipeline_multi_impl<O...>(e, std::forward<FuncIn>(in), std::forward<Stages>(sgs) ...);
  }
}



/// Runs a pipeline pattern with an intial function and a set of stages
/// FuncIn: Initial functor type.
/// Stages: intermediate and last stages
template <typename FuncIn, typename ...Stages>
void pipeline(polymorphic_execution & e, FuncIn && in, Stages && ... sgs) 
{
  pipeline_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(e, std::forward<FuncIn>(in), std::forward<Stages>(sgs) ...);
}




} // end namespace grppi

#endif
