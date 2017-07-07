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

#ifndef GRPPI_POLY_DIVIDEANDCONQUER_H
#define GRPPI_POLY_DIVIDEANDCONQUER_H

#include "common/polymorphic_execution.h"
#include "common/support.h"

namespace grppi{

template <typename Input, typename DivFunc,
          typename Operation, typename MergeFunc>
typename std::result_of<Operation(Input)>::type divide_and_conquer_multi_impl(polymorphic_execution & e, 
          Input & problem, DivFunc && divide, Operation && op, MergeFunc && merge)
{
  return {};
}



template <typename E, typename ... O,
          typename Input, typename DivFunc,
          typename Operation, typename MergeFunc,
          internal::requires_execution_not_supported<E> = 0>
typename std::result_of<Operation(Input)>::type divide_and_conquer_multi_impl(polymorphic_execution & e, 
          Input & problem, DivFunc && divide, Operation && op, MergeFunc && merge) 
{
  return divide_and_conquer_multi_impl<O...>(e, problem,  
      std::forward<DivFunc>(divide), std::forward<Operation>(op),
      std::forward<MergeFunc>(merge) );
}



template <typename E, typename ... O,
          typename Input, typename DivFunc,
          typename Operation, typename MergeFunc,
          internal::requires_execution_supported<E> = 0>
typename std::result_of<Operation(Input)>::type divide_and_conquer_multi_impl(polymorphic_execution & e, 
          Input & problem, DivFunc && divide, Operation && op, MergeFunc && merge) 
{
  if (typeid(E) == e.type()) {
    return divide_and_conquer(*e.execution_ptr<E>(), 
        problem, 
        std::forward<DivFunc>(divide), std::forward<Operation>(op),
        std::forward<MergeFunc>(merge));
  }
  else {
    return divide_and_conquer_multi_impl<O...>(e, problem,  
        std::forward<DivFunc>(divide), std::forward<Operation>(op),
        std::forward<MergeFunc>(merge));
  }
}


/// Runs a divide_and_conquer pattern with an initial problem, 
/// a divide function, an operation function and a merge function.
/// Input: input problem.
/// DivFunc: Division functor type.
/// Operation: Operation functor type.
/// MergeFunc: Merge functor type.
template <typename Input, typename DivFunc,
          typename Operation, typename MergeFunc>
typename std::result_of<Operation(Input)>::type divide_and_conquer(polymorphic_execution & e, 
          Input & problem, DivFunc && divide, Operation && op, MergeFunc && merge) 
{
  return divide_and_conquer_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(e, problem,  
      std::forward<DivFunc>(divide), std::forward<Operation>(op),
      std::forward<MergeFunc>(merge));
}

} // end namespace grppi

#endif
