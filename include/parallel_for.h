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

#ifndef GRPPI_PARALLEL_FOR_H
#define GRPPI_PARALLEL_FOR_H

#include <utility>

#include "common/execution_traits.h"
#include "common/iterator_traits.h"

namespace grppi {

/** 
\addtogroup task_patterns
@{
\defgroup parallel_for_pattern Parallel_for pattern
\brief Interface for applyinng the \ref md_parallel_for.
@{
*/

/**
\brief Invoke \ref md_parallel_for on a range of indexes

\tparam IndexType type of the of the index.
\tparam Transformer Callable object type for the transformation.
\param ex Sequential execution policy object.
\param first Initial value of the index.
\param last End value of the index.
\param step Increment of the index on each transformation call.
\param transform_op Transformation callable object.
\note The sequential_execution object acts exclusively as a tag type.
*/
template <typename Execution, typename IndexType, typename Transformer>
void parallel_for(Execution &ex, IndexType first, IndexType last,
    IndexType step, Transformer transform_op) 
{
  static_assert(supports_parallel_for<Execution>(),
      "map not supported on execution type");
  ex.parallel_for(first,last,step,std::forward<Transformer>(transform_op));
}

/**
@}
@}
*/
}

#endif
