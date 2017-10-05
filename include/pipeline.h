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

#ifndef GRPPI_PIPELINE_H
#define GRPPI_PIPELINE_H

#include <utility>

#include "common/callable_traits.h"
#include "common/execution_traits.h"
#include "common/pipeline_pattern.h"
#include "common/patterns.h"

namespace grppi {

/**
\addtogroup stream_patterns
@{
\defgroup pipeline_pattern Pipeline pattern
\brief Interface for applyinng the \ref md_pipeline
@{
*/


#include <iostream>
#include <utility>

//TODO: Check is is a function/functor object -> it doesn't work with overloaded functions
template <typename F>
using requires_no_execution =
  typename std::enable_if_t<!is_no_pattern<std::decay_t<F>> || internal::is_callable_type<F>::value, int>;


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
          requires_execution_supported<Execution> = 0>
void pipeline(
    const Execution & ex, 
    Generator && generate_op, 
    Transformers && ... transform_ops) 
{
  static_assert(supports_pipeline<Execution>(),
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
template <typename Generator, typename ... Transformers,
          requires_no_execution<Generator> = 0>
auto pipeline(
    Generator && generate_op,
    Transformers && ... transform_ops)
{
    return pipeline_t<Generator,Transformers...> (
        std::forward<Generator>(generate_op),
        std::forward<Transformers>(transform_ops)...);
}

/**
@}
@}
*/

}

#endif
