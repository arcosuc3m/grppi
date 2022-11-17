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
#ifndef GRPPI_CONTEXT_H
#define GRPPI_CONTEXT_H

#include <utility>

#include "grppi/common/execution_traits.h"

#include "grppi/common/context.h"

namespace grppi {

/** 
\addtogroup stream_patterns
@{
\defgroup context Context
\brief Interface for defining a new \ref md_context to run a given function or pattern.
@{
*/

/**
\brief Define a new \ref md_context on a data stream 
that can be composed in other streaming patterns.
\tparam Execution Execution policy type.
\tparam Transformer Callable type for the transformation operation.
\param ex Execution policy object.
\param transform_op Transformer operation.
*/
template <typename ExecutionPolicy, typename Transformer>
auto run_with(ExecutionPolicy & ex, Transformer && transform_op)
{
    static_assert(supports_context<ExecutionPolicy>(),
        "context not supported on execution type");
   return context_t<ExecutionPolicy,Transformer>{ex,
       std::forward<Transformer>(transform_op)};
}

/**
@}
@}
*/

}

#endif
