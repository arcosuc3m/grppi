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

#ifndef GRPPI_SPLIT_JOIN_H
#define GRPPI_SPLIT_JOIN_H

#include "common/split_join_pattern.h"

namespace grppi {

/** 
\addtogroup stream_patterns
@{
\defgroup farm_pattern Farm pattern
\brief Interface for applyinng the \ref md_farm.
@{
*/

/**
\brief Invoke \ref md_farm on a data stream 
that can be composed in other streaming patterns.
\tparam Execution Execution policy type.
\tparam Transformer Callable type for the transformation operation.
\param ex Execution policy object.
\param transform_op Transformer operation.
*/
template <typename Policy, typename ... Transformers>
auto split_join(Policy policy, Transformers && ... transforms)
{
   return split_join_t<Policy, Transformers ...>{
       policy, std::forward<Transformers>(transforms)...};
}

/**
@}
@}
*/

}

#endif
