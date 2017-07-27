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
execution with a generator, a predicate, a consumer and a transformer.
\tparam Generator Callable type for the generation operation.
\tparam Predicate Callable type for the predicate operation.
\tparam Consumer Callable type for the consume operation.
\tparam Transformer Callable type for the transformer operations.
\param generate_op Generator operation.
\param predicate_op Predicate operation.
\param consume_op Consumer operation.
\param transform_op Transformer operation.
*/
template<typename Generator, typename Transformer, typename Predicate, typename Consumer>
void repeat_until(sequential_execution, Generator && generate_op, Transformer && transform_op, Predicate && predicate_op, Consumer && consume_op){
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


template<typename Generator, typename Transformer, typename Predicate, typename Consumer>
void repeat_until(sequential_execution &ex, Generator && generate_op, farm_info<sequential_execution, Transformer> & farm, Predicate && predicate_op, Consumer && consume_op){
  repeat_until(ex, std::forward<Generator>( generate_op ), std::forward<farm_info<sequential_execution, Transformer> &&>( farm ), 
          std::forward<Predicate>( predicate_op), std::forward< Consumer >( consume_op ) );
}


/**
\brief Invoke \ref md_stream-iteration on a data stream with sequential 
execution with a generator, a predicate, a consumer and a farm as a transformer.
\tparam Generator Callable type for the generation operation.
\tparam Predicate Callable type for the predicate operation.
\tparam Consumer Callable type for the consume operation.
\tparam Transformer Callable type for the transformer operations.
\param generate_op Generator operation.
\param predicate_op Predicate operation.
\param consume_op Consumer operation.
\param farm Composed farm object.
*/
template<typename Generator, typename Transformer, typename Predicate, typename Consumer>
void repeat_until(sequential_execution &ex, Generator && generate_op, farm_info<sequential_execution, Transformer> && farm, Predicate && predicate_op, Consumer && consume_op){
  for(;;) {
    auto item = generate_op();       
    if (!item) break;
    auto val = *item;
    do {
      val = farm.task(val);
    } while (!predicate_op(val));
    consume_op(val);
  }
}

template<typename Generator, typename Predicate, typename Consumer, typename ...Stages>
void repeat_until(sequential_execution &ex, Generator && generate_op, pipeline_info<sequential_execution, Stages...> & pipe, Predicate && predicate_op, Consumer && consume_op){
  repeat_until(ex, std::forward<Generator>(generate_op), std::forward<pipeline_info<sequential_execution, Stages...> &&>( pipe ), std::forward<Predicate>(predicate_op), std::forward<Consumer>( consume_op ));
}

/**
\brief Invoke \ref md_stream-iteration on a data stream with sequential 
execution with a generator, a predicate, a consumer and a pipeline as a transformer.
\tparam Generator Callable type for the generation operation.
\tparam Predicate Callable type for the predicate operation.
\tparam Consumer Callable type for the consume operation.
\tparam Transformer Callable type for the transformer operations.
\param generate_op Generator operation.
\param predicate_op Predicate operation.
\param consume_op Consumer operation.
\param pipe Composed pipeline object.
*/
template<typename Generator, typename Predicate, typename Consumer, typename ...Stages>
void repeat_until(sequential_execution &ex, Generator && generate_op, pipeline_info<sequential_execution, Stages...> && pipe, Predicate && predicate_op, Consumer && consume_op){
  for (;;) {
    auto item = generate_op();
    if (!item) break; 
    auto val = *item;
    do {
      val = composed_pipeline<typename std::result_of<Generator()>::type::value_type,0,Stages...>(val,std::forward<pipeline_info<sequential_execution, Stages...>>(pipe) );
    } while (!predicate_op(val));
    consume_op(val);
  }
}

}

#endif
