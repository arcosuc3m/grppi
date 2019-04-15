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
#ifndef GRPPI_PIPELINE_H
#define GRPPI_PIPELINE_H

#include <utility>

#include "grppi/common/callable_traits.h"
#include "grppi/common/execution_traits.h"
#include "grppi/common/pipeline_pattern.h"

namespace grppi {

/**
\addtogroup stream_patterns
@{
\defgroup pipeline_pattern Pipeline pattern
\brief Interface for applyinng the \ref md_pipeline
@{
*/

/**
\brief Invoke \ref md_pipeline on a data stream.
\tparam Execution Execution type.
\tparam Generator Callable type for the stream generator.
\tparam Transformers Callable type for each transformation stage.
\param ex Execution policy object.
\param generate_op Generator operation.
\param trasnform_ops Transformation operations for each stage.
*/
template <typename Execution, typename Generator, typename ... Transformers,
          requires_execution_supported<std::decay_t<Execution>> = 0>
void pipeline(
    const Execution & ex, 
    Generator && generate_op, 
    Transformers && ... transform_ops) 
{
  static_assert(supports_pipeline<std::decay_t<Execution>>(),
      "pipeline pattern is not supported by execution type");
  ex.pipeline(std::forward<Generator>(generate_op),
      std::forward<Transformers>(transform_ops)...);
}

/**
\brief Build a composable \ref md_pipeline representation
that can be composed in other streaming patterns.
\tparam Execution Execution policy type.
\tparam Transformer Callable type for first transformation stage.
\tparam MoreTransformers Callable type for each additional transformation stage.
\param ex Execution policy object.
\param tranform_op First stage transformation operation
\param more_trasnform_ops Transformation operations for each additional stage.
*/
template <typename Transformer, typename ... Transformers,
          requires_execution_not_supported<std::decay_t<Transformer>> = 0>
auto pipeline(
    Transformer && transform_op,
    Transformers && ... transform_ops)
{
    return pipeline_t<Transformer, Transformers...>(
        std::forward<Transformer>(transform_op),
        std::forward<Transformers>(transform_ops)...);
}
/**
@}
@}
*/

}

#endif
