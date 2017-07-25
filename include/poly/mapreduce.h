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

#ifndef GRPPI_POLY_MAPREDUCE_H
#define GRPPI_POLY_MAPREDUCE_H

#include "../common/support.h"
#include "polymorphic_execution.h"

namespace grppi{

template <typename InputIt, typename Transformer, typename Identity, typename Combiner>
Identity map_reduce_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
  Identity identity, Transformer && transform_op, Combiner && combine_op) 
{
  return {};
}

template <typename E, typename ... O,
          typename InputIt, typename Transformer, typename Identity, typename Combiner,
          internal::requires_execution_not_supported<E> = 0>
Identity map_reduce_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, 
  Identity identity, Transformer && transform_op, Combiner && combine_op) 
{
  return map_reduce_multi_impl<O...>(e, first, last, identity, std::forward<Transformer>(transform_op), 
    std::forward<Combiner>(combine_op));
}

template <typename E, typename ... O,
          typename InputIt, typename Transformer, typename Identity, typename Combiner,
          internal::requires_execution_supported<E> = 0>
Identity map_reduce_multi_impl(polymorphic_execution & e, InputIt first, InputIt last, Identity identity,
  Transformer && transform_op, Combiner && combine_op) 
{
  if (typeid(E) == e.type()) {
    return map_reduce(*e.execution_ptr<E>(), 
        first, last, identity, std::forward<Transformer>(transform_op), 
        std::forward<Combiner>(combine_op));
  }
  else {
    return map_reduce_multi_impl<O...>(e, first, last, identity, std::forward<Transformer>(transform_op), 
        std::forward<Combiner>(combine_op));
  }
}

/**
\addtogroup mapreduce_pattern
@{
*/

/**
\addtogroup mapreduce_pattern_poly Polymorphic map/reduce pattern
\brief Polymorphic implementation of the \ref md_map-reduce pattern.
@{
*/

/**
\brief Invoke [map/reduce pattern](\ref md_map-reduce) on a data sequence with 
polymorphic execution.
\tparam InputIt Iterator type used for the input sequence.
\tparam Result Result type of the reduction.
\tparam Transformer Callable type for the transformation operation.
\tparam Combiner Callable type for the combination operation of the reduction.
\param ex Polymorphic execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param identity Identity value for the combination operation.
\param transf_op Transformation operation.
\param combine_op Combination operation.
\return Result of the map/reduce operation.
*/
template <typename InputIt, typename Transformer, typename Result, 
          typename Combiner>
Result map_reduce(polymorphic_execution & ex, 
                  InputIt first, InputIt last, Result identity,
                  Transformer && transform_op, 
                  Combiner && combine_op) 
{
  return map_reduce_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(ex, first, last, identity, std::forward<Transformer>(transform_op), 
        std::forward<Combiner>(combine_op));
}

/**
@}
@}
*/


} // end namespace grppi

#endif
