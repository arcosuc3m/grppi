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

#ifndef GRPPI_SEQ_STREAM_FILTER_H
#define GRPPI_SEQ_STREAM_FILTER_H

namespace grppi{

/** 
\addtogroup filter_pattern
@{
*/

/**
\addtogroup filter_pattern_seq Sequential filter pattern.
\brief Sequential implementation fo the \ref md_stream-filter pattern.
@{
*/

/**
\brief Invoke [stream filter pattern](@ref md_stream-filter pattern) on a data
sequence with sequential execution policy.
\tparam Generator Callable type for value generator.
\tparam Predicate Callable type for filter predicate.
\tparam Consumer Callable type for value consumer.
\param ex Sequential execution policy object.
\param generate_op Generator callable object.
\param predicate_op Predicate callable object.
\param consume_op Consumer callable object.
*/
template <typename Generator, typename Predicate, typename Consumer>
void stream_filter(sequential_execution, Generator generate_op, 
                   Predicate predicate_op, Consumer consume_op) 
{
  for (;;) {
    auto item = generate_op();
    if (!item) break;
    if (predicate_op(*item)) {
      consume_op(*item);
    }
  }
}

/**
@}
@}
*/

}

#endif
