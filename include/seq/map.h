/*
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

#ifndef GRPPI_SEQ_MAP_H
#define GRPPI_SEQ_MAP_H

#include "sequential_execution.h"

namespace grppi{

/**
\addtogroup map_pattern
@{
*/

/**
\addtogroup map_pattern_seq Sequential map pattern
\brief Sequential implementation of the \ref map-pattern
@{
*/

/**
\brief Invoke [map pattern](@ref map-pattern) on a data sequence with sequential
execution.
\tparam InputIt Iterator type used for input sequence.
\tparam OtuputIt Iterator type used for the output sequence.
\tparam Transformer Callable type for the transformation operation.
\param ex Sequential execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param first_out Iterator to first elemento of the output sequence.
\param transf_op Transformation operation.
*/
template <typename InputIt, typename OutputIt, typename Transformer>
void map(sequential_execution & ex, 
         InputIt first, InputIt last, OutputIt first_out, 
         Transformer && transf_op) 
{
  while(first != last) {
    *first_out = transf_op(*first);
    first++;
    first_out++;
  }
}

/**
\brief Invoke [map pattern](@ref map-pattern) on a data sequence with sequential
execution.
\tparam InputIt Iterator type used for input sequence.
\tparam OtuputIt Iterator type used for the output sequence.
\tparam Transformer Callable type for the transformation operation.
\tparam OtherInputIts Iterator types used for additional input sequences.
\param ex Sequential execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param first_out Iterator to first elemento of the output sequence.
\param transf_op Transformation operation.
\param more_firsts Additional iterators with first elements of additional sequences.
*/
template <typename InputIt, typename OutputIt, typename Transformer,
          typename ... OtherInputIts>
void map(sequential_execution & ex, 
         InputIt first, InputIt last, OutputIt first_out, 
         Transformer && transf_op, 
         OtherInputIts ... other_firsts) 
{
  while( first != last ) {
    *first_out = transf_op(*first, *other_firsts...);
    advance_iterators(other_firsts...);
    first++;
    first_out++;
  }
}

/**
@}
@}
*/
}
#endif
