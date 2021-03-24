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
#ifndef GRPPI_STREAM_FILTER_H
#define GRPPI_STREAM_FILTER_H

#include "grppi/common/patterns.h"

namespace grppi {

/** 
\addtogroup stream_patterns
@{
\defgroup filter_pattern Stream filter pattern
\brief Interface for applyinng the \ref md_stream-filter.
@{
*/

/**
\brief Invoke \ref md_stream-filter on a data stream
that can be composed in other streaming patterns.
This function keeps in the stream only those items
that satisfy the predicate.
\tparam Execution Execution policy.
\tparam Predicate Callable type for filter predicate.
\param ex Execution policy object.
\param predicate_op Predicate callable object.
*/
  template<typename Predicate>
  auto keep(Predicate && predicate_op)
  {
    return filter_t<Predicate>{std::forward<Predicate>(predicate_op), true};
  }

/**
\brief Invoke \ref md_stream-filter on a data stream
that can be composed in other streaming patterns.
This function discards from the stream those items
that satisfy the predicate.
\tparam Execution Execution policy.
\tparam Predicate Callable type for filter predicate.
\param ex Execution policy object.
\param predicate_op Predicate callable object.
*/
  template<typename Predicate>
  auto discard(Predicate && predicate_op)
  {
    return filter_t<Predicate>{std::forward<Predicate>(predicate_op), false};
  }

/**
@}
@}
*/

}

#endif
