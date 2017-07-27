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

#ifndef GRPPI_STREAM_REDUCE_H
#define GRPPI_STREAM_REDUCE_H

#include "seq/stream_reduce.h"
#include "native/stream_reduce.h"
#include "omp/stream_reduce.h"
#include "tbb/stream_reduce.h"
#include "poly/stream_reduce.h"

#include "common/patterns.h"

namespace grppi {

/** 
\addtogroup stream_patterns
@{
\defgroup stream_reduce_pattern Stream reduce pattern
\brief Interface for applying the \ref md_stream-reduce.
@{
*/

/**
\brief Invoke \ref md_stream-reduce on a stream
that can be composed in other streaming patterns.
\tparam Identity Type of the identity value used by the combiner.
\tparam Combiner Callable type used for data items combination.
\param ex Sequential execution policy object.
\param window_size Number of consecutive items to be reduced.
\param offset Number of items after of which a new reduction is started.
\param identity Identity value for the combination.
\param combine_op Combination operation.

*/
template <typename Execution, typename Identity, typename Combiner>
auto stream_reduce(Execution & ex, 
                   int window_size, int offset, 
                   Identity identity, 
                   Combiner && combine_op)
{
   return reduction_info<Execution, Combiner, Identity>(ex, 
       window_size, offset, identity, 
       std::forward<Combiner>(combine_op));
}

/**
@}
@}
*/

}

#endif
