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

#ifndef GRPPI_MAP_H
#define GRPPI_MAP_H

#include <utility>

#include "common/execution_traits.h"
#include "common/iterator_traits.h"

namespace grppi {

/** 
\addtogroup data_patterns
@{
\defgroup map_pattern Map pattern
\brief Interface for applyinng the \ref md_map.
@{
*/

/**
\brief Invoke \ref md_map on a data sequence.
\tparam InputIt Iterator type used for input sequence.
\tparam OtuputIt Iterator type used for the output sequence.
\tparam Transformer Callable type for the transformation operation.
\param ex Sequential execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param first_out Iterator to first elemento of the output sequence.
\param transform_op Transformation operation.
\note The sequential_execution object acts exclusively as a tag type.
*/
template <typename Execution, typename InputIt, typename OutputIt, 
          typename Transformer,
          requires_iterator<InputIt> = 0,
          requires_iterator<OutputIt> = 0>
void map(const Execution & ex, 
         InputIt first, InputIt last, OutputIt first_out, 
         Transformer transform_op) 
{
  static_assert(supports_map<Execution>(),
      "map not supported on execution type");
  ex.map(make_tuple(first), first_out,
      std::distance(first, last), transform_op);
}

/**
\brief Invoke \ref md_map on a data sequence.
\tparam InputIt Iterator type used for input sequence.
\tparam OtuputIt Iterator type used for the output sequence.
\tparam Transformer Callable type for the transformation operation.
\tparam OtherInputIts Iterator types used for additional input sequences.
\param ex Sequential execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param first_out Iterator to first elemento of the output sequence.
\param transform_op Transformation operation.
\param more_firsts Additional iterators with first elements of additional sequences.
\note The sequential_execution object acts exclusively as a tag type.
*/
template <typename Execution, typename InputIt, typename OutputIt, typename Transformer,
          typename ... OtherInputIts,
          requires_iterator<InputIt> = 0,
          requires_iterator<OutputIt> = 0>
void map(const Execution & ex, 
         InputIt first, InputIt last, OutputIt first_out, 
         Transformer transform_op, 
         OtherInputIts ... other_firsts) 
{
  static_assert(supports_map<Execution>(),
      "map not supported on execution type");
  ex.map(make_tuple(first,other_firsts...), first_out,
      std::distance(first,last), transform_op);
}

/**
@}
@}
*/
}

#endif
