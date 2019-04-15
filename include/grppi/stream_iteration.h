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
#ifndef GRPPI_STREAM_ITERATION_H
#define GRPPI_STREAM_ITERATION_H

#include "grppi/common/iteration_pattern.h"

namespace grppi {

/**
\addtogroup stream_patterns
@{
\defgroup stream_iteration_pattern Stream iteration pattern
\brief Interface for applyinng the \ref md_stream-iteration.
@}
*/

template <typename Transformer, typename Predicate>
auto repeat_until(
    Transformer && transform_op,
    Predicate && predicate_op)
{
  return iteration_t<Transformer,Predicate>(
      std::forward<Transformer>(transform_op),
      std::forward<Predicate>(predicate_op));
}

}

#endif
