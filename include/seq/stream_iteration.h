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

#ifndef GRPPI_SEQ_STREAM_ITERATION_H
#define GRPPI_SEQ_STREAM_ITERATION_H

#include "sequential_execution.h"

namespace grppi{

/**
\addtogroup stream_iteration_pattern
@{
\addtogroup stream_iteration_pattern_sequential Sequential stream iteration pattern
\brief Sequential implementation of the \ref md_stream-iteration.
@{
*/

/**
\brief Invoke \ref md_stream-iteration on a data stream with sequential 
execution with a generator, a transformer, a predicate, and a consumer.
\tparam Generator Callable type for the generation operation.
\tparam Transformer Callable type for the transformer operations.
\tparam Predicate Callable type for the predicate operation.
\tparam Consumer Callable type for the consume operation.
\param generate_op Generator operation.
\param predicate_op Predicate operation.
\param consume_op Consumer operation.
\param transform_op Transformer operation.
*/
template<typename Generator, typename Transformer, typename Predicate, 
         typename Consumer>
void repeat_until(sequential_execution, 
                  Generator generate_op, Transformer && transform_op, 
                  Predicate predicate_op, Consumer consume_op)
{
  for(;;) {
    auto item = generate_op();
    if (!item) break;
    auto val = *item;
    do {
      val = transform_op(val);
    } while (!predicate_op(val));
    consume_op(val);
  }
}

/**
\brief Invoke \ref md_stream-iteration on a data stream with sequential 
execution with a generator, a farm as transformer, a predicate, and a consumer.
\tparam Generator Callable type for the generation operation.
\tparam Transformer Callable type for the transformer operations.
\tparam Predicate Callable type for the predicate operation.
\tparam Consumer Callable type for the consume operation.
\param ex Sequential execution policy object.
\param generate_op Generator operation.
\param farm_obj Composed farm object.
\param predicate_op Predicate operation.
\param consume_op Consumer operation.
\note This version takes the far by l-value reference.
*/
template<typename Generator, typename Transformer, typename Predicate, 
         typename Consumer>
void repeat_until(sequential_execution & ex, 
                  Generator && generate_op, 
                  farm_info<sequential_execution, Transformer> & farm_obj, 
                  Predicate && predicate_op, Consumer && consume_op)
{
  repeat_until(ex, 
      std::forward<Generator>(generate_op), 
      std::forward<farm_info<sequential_execution,Transformer> &&>(farm_obj), 
      std::forward<Predicate>(predicate_op), 
      std::forward<Consumer>(consume_op));
}


/**
\brief Invoke \ref md_stream-iteration on a data stream with sequential 
execution with a generator, a farm as transformer, a predicate, and a consumer.
\tparam Generator Callable type for the generation operation.
\tparam Transformer Callable type for the transformer operations.
\tparam Predicate Callable type for the predicate operation.
\tparam Consumer Callable type for the consume operation.
\param generate_op Generator operation.
\param farm_obj Composed farm object.
\param predicate_op Predicate operation.
\param consume_op Consumer operation.
\note This version takes the far by r-value reference.
*/
template<typename Generator, typename Transformer, typename Predicate, 
         typename Consumer>
void repeat_until(sequential_execution & ex, 
                  Generator generate_op, 
                  farm_info<sequential_execution, Transformer> && farm_obj, 
                  Predicate predicate_op, 
                  Consumer consume_op)
{
  for(;;) {
    auto item = generate_op();       
    if (!item) break;
    auto val = *item;
    do {
      val = farm_obj.task(val);
    } while (!predicate_op(val));
    consume_op(val);
  }
}

/**
\brief Invoke \ref md_stream-iteration on a data stream with sequential 
execution with a generator, a pipeline as transformer, a predicate, and a consumer.
\tparam Generator Callable type for the generation operation.
\tparam Transformer Callable type for the transformer operations.
\tparam Predicate Callable type for the predicate operation.
\tparam Consumer Callable type for the consume operation.
\param generate_op Generator operation.
\param pipe_obj Composed pipe object.
\param predicate_op Predicate operation.
\param consume_op Consumer operation.
\note This version takes the far by l-value reference.
*/
template<typename Generator, typename Predicate, typename Consumer, 
         typename ... Transformers>
void repeat_until(sequential_execution & ex, 
                  Generator && generate_op, 
                  pipeline_info<sequential_execution, 
                      Transformers...> & pipe_obj, 
                  Predicate && predicate_op, Consumer && consume_op)
{
  using namespace std;
  repeat_until(ex, 
      forward<Generator>(generate_op), 
      forward<pipeline_info<sequential_execution, 
              Transformers...> &&>(pipe), 
      forward<Predicate>(predicate_op), 
      forward<Consumer>(consume_op));
}

/**
\brief Invoke \ref md_stream-iteration on a data stream with sequential 
execution with a generator, a pipeline as transformer, a predicate, and a consumer.
\tparam Generator Callable type for the generation operation.
\tparam Predicate Callable type for the predicate operation.
\tparam Consumer Callable type for the consume operation.
\tparam Transformers Callable types for the transformers operations.
\param generate_op Generator operation.
\param pipe_obj Composed pipeline object.
\param predicate_op Predicate operation.
\param consume_op Consumer operation.
\note This version takes the far by r-value reference.
*/
template<typename Generator, typename Predicate, typename Consumer, 
         typename ...Transformers>
void repeat_until(sequential_execution & ex, 
                  Generator generate_op, 
                  pipeline_info<sequential_execution, Transformers...> && pipe, 
                  Predicate predicate_op, Consumer consume_op)
{
  using namespace std;
  for (;;) {
    auto item = generate_op();
    if (!item) break; 
    auto val = *item;
    do {
      using generated_type = typename result_of<Generator()>::type;
      using generated_value_type = typename generated_type::value_type;
      using pipeline_info_type = pipeline_info<sequential_execution,
          Transformers...>;
      val = composed_pipeline<generated_value_type,0,Transformers...>(val, 
          forward<pipeline_info_type>(pipe));
    } while (!predicate_op(val));
    consume_op(val);
  }
}

}

#endif
