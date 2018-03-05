/**
* @version		GrPPI v0.3
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

#ifndef GRPPI_REDUCE_H 
#define GRPPI_REDUCE_H

#include <utility>

#include "common/iterator_traits.h"
#include "common/execution_traits.h"

namespace grppi {

/** 
\addtogroup data_patterns
@{
\defgroup reduce_pattern Reduce pattern
\brief Interface for applyinng the \ref md_reduce.
@{
*/

/**
\brief Invoke \ref md_reduce with identity value
on a data sequence with sequential execution.
\tparam Execution Execution type.
\tparam InputIt Iterator type used for input sequence.
\tparam Result Type for the identity value.
\tparam Combiner Callable type for the combiner operation.
\param ex Execution policy object.
\param first Iterator to the first element in the input sequence.
\param size Size of the input sequence to be process.
\param identity Identity value for the combiner operation.
\param combiner_op Combiner operation for the reduction.
\return The result of the reduction.
*/
template <typename Execution, typename InputIt, typename Result, typename Combiner,
    requires_iterator<InputIt> = 0>
auto reduce(const Execution & ex,
            InputIt first, std::size_t size,
            Result && identity,
            Combiner && combine_op)
{
  static_assert(supports_reduce<Execution>(),
                "reduce not supported on execution type");
  return ex.reduce(first, size,
                   std::forward<Result>(identity), std::forward<Combiner>(combine_op));
}

/**
\brief Invoke \ref md_reduce with identity value
on a data sequence with sequential execution.
\tparam Execution Execution type.
\tparam InputIt Iterator type used for input sequence.
\tparam Result Type for the identity value.
\tparam Combiner Callable type for the combiner operation.
\param ex Execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param identity Identity value for the combiner operation.
\param combiner_op Combiner operation for the reduction.
\return The result of the reduction.
*/
template <typename Execution, typename InputIt, typename Result, typename Combiner,
          requires_iterator<InputIt> = 0>
auto reduce(const Execution & ex, 
            InputIt first, InputIt last, 
            Result && identity,
            Combiner && combine_op)
{
  static_assert(supports_reduce<Execution>(),
      "reduce not supported on execution type");
  return ex.reduce(first, std::distance(first,last), 
      std::forward<Result>(identity), std::forward<Combiner>(combine_op));
}

/**
@}
@}
*/
}

#endif
