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

#include "../common/support.h"
#include "polymorphic_execution.h"

namespace grppi{

template <typename InputIt, typename Identity, typename Combiner>
auto reduce_multi_impl(polymorphic_execution & ex, 
                       InputIt first, InputIt last, 
                       Identity identity,
                       Combiner &&combine_op) -> decltype(combine_op(*first,*first))
{
  return {};
}



template <typename E, typename ... O,
          typename InputIt, typename Identity, typename Combiner,
          internal::requires_execution_not_supported<E> = 0>
auto reduce_multi_impl(polymorphic_execution & ex, 
                       InputIt first, InputIt last, 
                       Identity identity,
                       Combiner && combine_op)
{
  return reduce_multi_impl<O...>(ex, first, last, identity, std::forward<Combiner>(combine_op));
}



template <typename E, typename ... O,
          typename InputIt, typename Identity, typename Combiner,
          internal::requires_execution_supported<E> = 0>
auto reduce_multi_impl(polymorphic_execution & ex, 
                       InputIt first, InputIt last, 
                       Identity identity,
                       Combiner &&combine_op)
{
  if (typeid(E) == ex.type()) {
    return reduce(*ex.execution_ptr<E>(), 
        first, last, identity, std::forward<Combiner>(combine_op));
  }
  else {
    return reduce_multi_impl<O...>(ex, first, last, identity, std::forward<Combiner>(combine_op));
  }
}

/**
\addtogroup reduce_pattern
@{
\addtogroup reduce_pattern_poly Polymorphic execution reduce pattern.
\brief Polymorphic implementation of \ref md_reduce.
@{
*/


/**
\brief Invoke \ref md_reduce with no identity value
on a data sequence with sequential execution.
\tparam InputIt Iterator type used for input sequence.
\tparam Identity Type for the identity value.
\tparam Combiner Callable type for the combiner operation.
\param ex Sequential execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param identity Identity value for the combiner operation.
\param combiner_op Combiner operation for the reduction.
*/
template <typename InputIt, typename Identity, typename Combiner>
auto reduce(polymorphic_execution & e, 
            InputIt first, InputIt last, 
            Identity identity, 
            Combiner &&combine_op)
{
  return reduce_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(e, first, last, identity, std::forward<Combiner>(combine_op));
}

/**
@}
@}
*/

} // end namespace grppi

#endif
