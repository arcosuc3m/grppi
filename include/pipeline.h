/*
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

#ifndef GRPPI_PIPELINE_H
#define GRPPI_PIPELINE_H


#include "common/callable_traits.h"

#include "common/patterns.h"
#include "seq/pipeline.h"
#include "native/pipeline.h"
#include "omp/pipeline.h"
#include "tbb/pipeline.h"
#include "poly/pipeline.h"

/**
\addtogroup stream_patterns
@{
\defgroup pipeline_pattern Pipeline pattern
\brief Interface for applyinng the \ref md_pipeline
@}
*/

namespace grppi {

/**
\addtogroup pipeline_pattern
@{
*/

/**
\brief Build a composable [pipeline pattern](@ref md_pipeline) representation
that can be inserted into another streaming pattern.
\tparam Execution Execution policy type.
\tparam Transformer Callable type for first transformation stage.
\tparam MoreTransformers Callable type for each additional transformation stage.
\param ex Execution policy object.
\param tranform_op First stage transformation operation
\param more_trasnform_ops Transformation operations for each additional stage.
*/
template <typename Execution, typename Transformer, 
          typename ... MoreTransformers,
          requires_arguments<Transformer> = 0>
pipeline_info<Execution,Transformer,MoreTransformers...> 
pipeline(Execution & ex, Transformer && transform_op, 
         MoreTransformers && ... more_transform_ops)
{
    return pipeline_info<Execution,Transformer, MoreTransformers...> (ex, 
        std::forward<Transformer>(transform_op), 
        std::forward<MoreTransformers>(more_transform_ops)...);
}



/**
@}
*/

}

#endif
