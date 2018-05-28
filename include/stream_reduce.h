/*
 * Copyright 2018 Universidad Carlos III de Madrid
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GRPPI_STREAM_REDUCE_H
#define GRPPI_STREAM_REDUCE_H

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
template <typename Identity, typename Combiner>
auto reduce(int window_size, int offset, 
                   Identity identity, 
                   Combiner && combine_op)
{
//  static_assert(std::is_same<Identity,typename std::result_of<Combiner(Identity,Identity)>::type>::value,
//                "reduce combiner should be homogeneous:T = op(T,T)");
  return reduce_t<Combiner,Identity>(
       window_size, offset, identity, 
       std::forward<Combiner>(combine_op));
}

/**
@}
@}
*/

}

#endif
