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

#ifndef GRPPI_TBB_PIPELINE_H
#define GRPPI_TBB_PIPELINE_H

#ifdef GRPPI_TBB

#include "parallel_execution_tbb.h"

#include <experimental/optional>

#include <tbb/tbb.h>

namespace grppi {

// TODO: Input could be only a template argument with no function argument.


//Last stage
template <typename Transformer, typename Input>
auto pipeline_impl(parallel_execution_tbb & ex, 
                   Input, 
                   Transformer && transform_op) 
{
    return tbb::make_filter<std::experimental::optional<Input>, void>(
        tbb::filter::serial_in_order,
        [&](std::experimental::optional<Input> s) { 
          if(s) transform_op(*s);
        } 
    );
}


//Intermediate stages
template <typename Transformer, typename ... MoreTransformers, typename Input>
auto pipeline_impl(parallel_execution_tbb & ex, 
                   Input in, 
                   farm_info<parallel_execution_tbb, Transformer> & farm_obj, 
                   MoreTransformers && ... more_transform_ops) 
{
  using farm_type = farm_info<parallel_execution_tbb,Transformer>;

  return pipeline_impl(ex, in, std::forward<farm_type>(farm_obj), 
      std::forward<MoreTransformers>(more_transform_ops)...);
}


template <typename Predicate, typename... MoreTransformers, typename Input>
auto pipeline_impl(parallel_execution_tbb & ex, 
                   Input in,
                   filter_info<parallel_execution_tbb,Predicate> & filter_obj, 
                   MoreTransformers && ... more_transform_ops) 
{
  using filter_type = filter_info<parallel_execution_tbb,Predicate>;

  return pipeline_impl(ex, in, std::forward<filter_type>(filter_obj), 
      std::forward<MoreTransformers>(more_transform_ops)...);
}

template <typename Predicate, typename... MoreTransformers, typename Input>
auto pipeline_impl(parallel_execution_tbb & ex, 
                   Input in,
                   filter_info<parallel_execution_tbb,Predicate> && filter_obj, 
                   MoreTransformers && ... more_transform_ops) 
{
  using namespace std;
  using optional_input_type = experimental::optional<Input>;

  return 
      tbb::make_filter<optional_input_type,optional_input_type>(
          tbb::filter::parallel,
          [&](optional_input_type val) -> optional_input_type { 
            return (val && filter_obj.task(*val)) ? 
              (val) :
              optional_input_type{};
          }
      ) 
      & 
      pipeline_impl(ex, in, 
          std::forward<MoreTransformers>(more_transform_ops)...);
}


template <typename Transformer, typename ... MoreTransformers, typename Input>
auto pipeline_impl(parallel_execution_tbb & ex, 
                   Input, 
                   farm_info<parallel_execution_tbb, Transformer> && farm_obj, 
                   MoreTransformers && ... more_transform_ops ) 
{
  using namespace std;
  using optional_input_type = experimental::optional<Input>;
  using output_type = typename result_of<Transformer(Input)>::type;
  using optional_output_type = experimental::optional<output_type>;

  return 
      tbb::make_filter<optional_input_type, optional_output_type>(
          tbb::filter::parallel,
          [&](optional_input_type val) -> optional_output_type {
          return (val) ? 
              farm_obj.task(*val) : 
              optional_output_type{};
          } 
      ) 
      &
      pipeline_impl(ex, output_type{}, 
          forward<MoreTransformers>(more_transform_ops)...);
}

template <typename Predicate, typename ... MoreTransformers, typename Input>
auto pipeline_impl(parallel_execution_tbb & ex, 
                   Input, 
                   Predicate &&  predicate_op, 
                   MoreTransformers && ... more_transform_ops) 
{
  using namespace std;
  using optional_input_type = experimental::optional<Input>;
  using output_type = typename result_of<Predicate(Input)>::type;
  using optional_output_type = experimental::optional<output_type>;

  return 
      tbb::make_filter<optional_input_type, optional_output_type>( 
          tbb::filter::serial_in_order,
          [&](optional_input_type val) -> optional_output_type {
            return (val) ? predicate_op(*val) : 
              optional_output_type{};
          }
      ) 
      & 
      pipeline_impl(ex, output_type{},      
          forward<MoreTransformers>(more_transform_ops)...);
}

/**
\addtogroup pipeline_pattern
@{
\addtogroup pipeline_pattern_tbb TBB parallel pipeline pattern
\brief TBB parallel implementation of the \ref md_pipeline. 
@{
*/

/**
\brief Invoke \ref md_pipeline on a data stream
with TBB parallel execution.
\tparam Generator Callable type for the stream generator.
\tparam Transformers Callable type for each transformation stage.
\param ex TBB parallel execution policy object.
\param generate_op Generator operation.
\param trasnform_ops Transformation operations for each stage.
\remark Generator shall be a zero argument callable type.
*/
template <typename Generator, typename ... Transformers,
          requires_no_arguments<Generator> = 0>
void pipeline(parallel_execution_tbb & ex, Generator generate_op, 
              Transformers && ... transform_ops) 
{
  using namespace std;
  using result_type = typename result_of<Generator()>::type;
  using output_value_type = typename result_type::value_type;
  using output_type = experimental::optional<output_value_type>;

  const auto this_filter = tbb::make_filter<void, output_type>(
      tbb::filter::serial_in_order, 
      [&](tbb::flow_control& fc) -> output_type {
        auto item =  generate_op();
        if (!item) { fc.stop(); }
        return (item) ? *item :output_type{};
      }
  );

  tbb::task_group_context context;
  tbb::parallel_pipeline(ex.tokens(), 
      this_filter 
      & 
      pipeline_impl(ex, output_value_type{}, 
          forward<Transformers>(transform_ops)...));
}

/**
@}
@}
*/

}

#endif

#endif
