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

#ifndef GRPPI_SEQ_FARM_H
#define GRPPI_SEQ_FARM_H

#include "sequential_execution.h"

namespace grppi{

/**
\addtogroup farm_pattern
@{
*/

/**
\addtogroup farm_pattern_seq Sequential farm pattern
Sequential implementation of the \ref md_farm.
@{
*/

/**
\brief Invoke [farm pattern](@ref md_farm) on a data stream with sequential
execution with a generator and a consumer.
\tparam Generator Callable type for the generation operation.
\tparam Consumer Callable type for the consume operation.
\param ex Sequential execution policy object.
\param generate_op Gnerator operation.
\param consume_op Consumer operation.
*/
template <typename Generator, typename Consumer>
void farm(sequential_execution ex, 
          Generator generate_op, Consumer consume_op) 
{
  for (;;) {
    auto item{generate_op()};
    if(!item) break;
    consume_op(*item);
  }
}

/**
\brief Invoke [farm pattern](@ref md_farm) on a data stream with sequential
execution with a generator, a transformer, and a comsumer.
\tparam Generator Callable type for the generation operation.
\tparam Transformer Callable type for the transformation operation.
\tparam Consumer Callable type for the consume operation.
\param ex Sequential execution policy object.
\param generate_op Generator operation.
\param transform_op Transformer operation.
\param consume_op Consumer operation.
*/
template <typename Generator, typename Transformer, typename Consumer>
void farm(sequential_execution ex, 
          Generator generate_op, Transformer transform_op, Consumer consume_op)
{
  for (;;) {
    auto item{generate_op()};
    if (!item) break;
    consume_op(transform_op(*item));
  }
}

}
#endif
