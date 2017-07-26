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

#ifndef GRPPI_POLY_PIPELINE_H
#define GRPPI_POLY_PIPELINE_H

#include "../common/support.h"
#include "polymorphic_execution.h"

namespace grppi{

template <typename Execution, typename Transformer,
          typename ... MoreTransformers,
          requires_arguments<Transformer> = 0>
pipeline_info<Execution,Transformer,MoreTransformers...>
transform_pipeline(Execution & ex, std::tuple<Transformer, MoreTransformers ...> && transform_ops)
{
    return pipeline_info<Execution,Transformer, MoreTransformers...> (ex,std::forward<std::tuple<Transformer,MoreTransformers...>>(transform_ops));
}




template <typename Generator, typename ... Transformers>
void pipeline_multi_impl(polymorphic_execution &, Generator &&, 
                         Transformers && ...) 
{

}

template <typename E, typename ... O,
          typename Generator, typename ... Transformers,
          internal::requires_execution_not_supported<E> = 0>
void pipeline_multi_impl(polymorphic_execution & ex, Generator && generate_op, 
                         Transformers && ... transform_ops) 
{
  pipeline_multi_impl<O...>(ex, std::forward<Generator>(generate_op), 
    std::forward<Transformers>(transform_ops) ...);
}

template <typename E, typename ... O,
          typename Generator, typename ... Transformers,
          internal::requires_execution_supported<E> = 0>
void pipeline_multi_impl(polymorphic_execution & ex, Generator && generate_op, 
                         Transformers && ... transform_ops) 
{
  if (typeid(E) == ex.type()) {
    pipeline(*ex.execution_ptr<E>(), 
      std::forward<Generator>(generate_op), 
      std::forward<Transformers>(transform_ops)...);
  }
  else {
    pipeline_multi_impl<O...>(ex, std::forward<Generator>(generate_op), 
        std::forward<Transformers>(transform_ops)...);
  }
}

/**
\addtogroup pipeline_pattern
@{
\addtogroup pipeline_pattern_poly Polymorphic pipeline pattern
\brief Polymorphic implementation of the \ref md_pipeline.
@{
*/

/**
\brief Invoke \ref md_pipeline on a data stream
with polymorphic execution.
\tparam Generator Callable type for the stream generator.
\tparam Transformers Callable type for each transformation stage.
\param ex Polymorphic execution policy object.
\param generate_op Generator operation.
\param trasnform_ops Transformation operations for each stage.
*/
template <typename Generator, typename ... Transformers,
          requires_no_arguments<Generator> = 0>
void pipeline(polymorphic_execution & ex, Generator && generate_op, 
              Transformers && ... transform_ops) 
{
  pipeline_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(ex, std::forward<Generator>(generate_op), 
      std::forward<Transformers>(transform_ops) ...);
}

/**
@}
@}
*/

} // end namespace grppi

#endif
