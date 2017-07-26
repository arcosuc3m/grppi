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

#ifndef GRPPI_NATIVE_STREAM_REDUCE_H
#define GRPPI_NATIVE_STREAM_REDUCE_H

#include <thread>

#include "parallel_execution_native.h"

namespace grppi{

/**
\addtogroup stream_reduce_pattern
@{
*/

/**
\addtogroup stream_reduce_pattern_native Native parallel stream reduce pattern
\brief Native parallel implementation of the \ref md_stream-reduce pattern.
@{
*/

/**
\brief Invoke [stream reduce pattern](@ref md_stream-reduce) on a stream with
native parallel execution.
\tparam Identity Type of the identity value used by the combiner.
\tparam Generator Callable type used for generating data items.
\tparam Combiner Callable type used for data items combination.
\tparam Consumer Callable type used for consuming data items.
\param ex Native parallel execution policy object.
\param window_size Number of consecutive items to be reduced.
\param offset Number of items after of which a new reduction is started.
\param identity Identity value for the combination.
\param generate_op Generation operation.
\param combine_op Combination operation.
\param consume_op Consume operation.
*/
template <typename Identity, typename Combiner, typename Consumer, 
          typename Generator>
void stream_reduce(parallel_execution_native & ex,
                   int window_size, int offset, Identity identity,
                   Generator && generate_op, Combiner && combine_op, 
                   Consumer && consume_op)
{
  using namespace std;
  using generated_type = typename result_of<Generator()>::type;
  using generated_value_type = typename generated_type::value_type;
  // TODO: Evaluate better structure than vector
  vector<generated_value_type> values;
  values.reserve(window_size);

  // TODO: Set generator and consumer in separate threads
  auto item = generate_op();
  for (;;) {
    while (item && values.size()!=window_size) {
      values.push_back(*item);
      item = generate_op();
    }
    if (values.size()>0) {
      auto reduced_value = reduce(ex, values.begin(), values.end(), identity,
          std::forward<Combiner>(combine_op));
      consume_op(reduced_value);
      if (item) {
        if (offset <= window_size) {
          values.erase(values.begin(), values.begin() + offset);
        }
        else {
          values.erase(values.begin(), values.end());
          auto diff = offset - window_size;
          while (diff > 0 && item) {
            item = generate_op();
            diff--;
          }
        }
      }
    }
    if (!item) break;
  }
}

/**
@}
@}
*/

}
#endif
