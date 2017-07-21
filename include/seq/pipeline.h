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

#ifndef GRPPI_SEQ_PIPELINE_H
#define GRPPI_SEQ_PIPELINE_H

#include "sequential_execution.h"
#include "common/pack_traits.h"

namespace grppi {

template <typename Input, int Index, typename ... MoreTransformers,
          internal::requires_index_last<Index,MoreTransformers...> = 0>
auto composed_pipeline(Input in, 
                       const pipeline_info<
                           sequential_execution,
                           MoreTransformers...> & pipeline_obj)
{
  return std::get<Index>(pipeline_obj.stages)(in);
}

template <typename Input, int Index, typename ... MoreTransformers,
          internal::requires_index_not_last<Index,MoreTransformers...> = 0>
auto composed_pipeline(Input in, 
                       const pipeline_info<
                           sequential_execution, 
                           MoreTransformers...> & pipeline_obj)
{
  return composed_pipeline<Input, Index+1, MoreTransformers...>(
      std::get<Index>(pipeline_obj.stages)(in),
      pipeline_obj);
}

//Last stage
template <typename Item, typename Transformer>
void pipeline_impl(sequential_execution &s, Item && item, 
                   Transformer && transform_op) 
{
  transform_op(std::forward<Item>(item));
}

//Filter stage
template <typename Item, typename Filter, typename... MoreTransformers>
void pipeline_impl(sequential_execution & ex, Item && item, 
                   const filter_info<sequential_execution,Filter> & filter_obj, 
                   MoreTransformers && ... more_transform_ops)
{
  pipeline_impl(ex, std::forward<Item>(item), 
      std::forward<filter_info<sequential_execution,Filter>>(filter_obj), 
      std::forward<MoreTransformers>(more_transform_ops)...);
}

template <typename Item, typename Transformer, typename... MoreTransformers>
void pipeline_impl(sequential_execution & ex, Item && item, 
                   filter_info<sequential_execution, Transformer> && transf_filter, 
                   MoreTransformers && ... more_transform_ops) 
{
  if(transf_filter.task(item))
    pipeline_impl(ex, std::forward<Item>(item),
        std::forward<MoreTransformers>(more_transform_ops)...);
}

//Farm stage
template <typename Item, typename Transformer, typename... MoreTransformers>
void pipeline_impl(sequential_execution & ex, Item && item, 
                   farm_info<sequential_execution, Transformer> & transf_farm, 
                   MoreTransformers && ... more_transform_ops) 
{
  pipeline_impl(ex, std::forward<Item>(item), 
      std::forward<farm_info<sequential_execution,Transformer>>(transf_farm), 
      std::forward<MoreTransformers>(more_transform_ops)... );
}

template <typename Item, typename Transformer, typename... MoreTransformers>
void pipeline_impl(sequential_execution & ex, Item && item, 
                   farm_info<sequential_execution, Transformer> && transform_farm, 
                   MoreTransformers && ... more_transform_ops) 
{
  pipeline_impl(ex, transform_farm.task(item), 
      std::forward<MoreTransformers>(more_transform_ops)...);
}



//Intermediate stages
template <typename Item, typename Transformer, typename ... MoreTransformers>
void pipeline_impl(sequential_execution & ex, Item && item, 
                   Transformer && transform_op, 
                   MoreTransformers && ... more_transform_ops) 
{
  pipeline_impl(ex, transform_op(item), 
      std::forward<MoreTransformers>(more_transform_ops)...);
}

/**
\addtogroup pipeline_pattern
@{
*/

/**
\addtogroup pipeline_pattern_seq Sequential pipeline pattern
\brief Sequential implementation of the \ref md_pipeline pattern
@{
*/

/**
\brief Invoke [pipeline pattern](@ref md_pipeline) on a data stream
with sequential execution.
\tparam Generator Callable type for the stream generator.
\tparam Transformers Callable type for each transformation stage.
\param ex Sequential execution policy object.
\param generate_op Generator operation.
\param trasnform_ops Transformation operations for each stage.
*/
template <typename Generator, 
          typename ... Transformers,
          typename = typename std::result_of<Generator()>::type> // TODO: Intention?
void pipeline(sequential_execution & ex, Generator && generator_op, 
              Transformers && ... transform_ops) 
{
  for (;;) {
    auto item = generator_op();
    if(!item) break;
    pipeline_impl(ex, *item, std::forward<Transformers>(transform_ops) ... );
  }
}

/**
@}
@}
*/
}

#endif
