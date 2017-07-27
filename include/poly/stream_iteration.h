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

#ifndef GRPPI_POLY_STREAM_ITERATOR_H
#define GRPPI_POLY_STREAM_ITERATOR_H

#include "polymorphic_execution.h"
#include "../common/support.h"

namespace grppi {

template<typename Generator, typename Predicate, typename Consumer,
         typename Transformer >
void repeat_until_multi_impl(polymorphic_execution & e, Generator && in,
      farm_info<polymorphic_execution,Transformer> && op, Predicate && predicate_op, Consumer && out)
{
}

template <typename E, typename ... O,
          typename Generator, typename Predicate, typename Consumer,
          typename Transformer ,
          internal::requires_execution_not_supported<E> = 0>
void repeat_until_multi_impl(polymorphic_execution & e,  Generator && in,
      pipeline_info<polymorphic_execution,Transformer> && op, Predicate && predicate_op, Consumer && out)
{
  repeat_until_multi_impl<O...>(e, std::forward<Generator>(in),
    std::forward<farm_info<polymorphic_execution,Transformer> >(op), std::forward<Predicate>(predicate_op),
    std::forward<Consumer>(out));
}

template <class E, typename ... O,
          typename Generator, typename Predicate, typename Consumer,
          typename Transformer ,
          internal::requires_execution_supported<E> = 0>
void repeat_until_multi_impl(polymorphic_execution & e, Generator && in,
      farm_info<polymorphic_execution,Transformer> && op, Predicate && predicate_op, Consumer && out)
{
  if (typeid(E) == e.type() && typeid(E) == op.exectype.type()) {
    auto & pipe_exec = op.exectype;
    auto & actual_exec = *pipe_exec. template execution_ptr<E>();
    auto transformed_farm = farm(actual_exec, std::forward<Transformer>(op.task));
    repeat_until(*e.execution_ptr<E>(),
        std::forward<Generator>(in),
        std::forward<farm_info<E,Transformer> >(transformed_farm), std::forward<Predicate>(predicate_op),
        std::forward<Consumer>(out));
  }
  else {
    repeat_until_multi_impl<O...>(e, std::forward<Generator>(in),
        std::forward<farm_info<polymorphic_execution,Transformer> >(op), std::forward<Predicate>(predicate_op),
        std::forward<Consumer>(out));
  }
}

template<typename Generator, typename Predicate, typename Consumer,
         typename ...MoreTransformers >
void repeat_until_multi_impl(polymorphic_execution & e, Generator && in,
      pipeline_info<polymorphic_execution,MoreTransformers...> && op, Predicate && predicate_op, Consumer && out)
{
}

template <typename E, typename ... O,
          typename Generator, typename Predicate, typename Consumer,
          typename ...MoreTransformers ,
          internal::requires_execution_not_supported<E> = 0>
void repeat_until_multi_impl(polymorphic_execution & e,  Generator && in,
      pipeline_info<polymorphic_execution,MoreTransformers...> && op, Predicate && predicate_op, Consumer && out)
{
  repeat_until_multi_impl<O...>(e, std::forward<Generator>(in),
    std::forward<pipeline_info<polymorphic_execution,MoreTransformers...> >(op), std::forward<Predicate>(predicate_op),
    std::forward<Consumer>(out));
}

template <class E, typename ... O,
          typename Generator, typename Predicate, typename Consumer,
          typename ...MoreTransformers ,
          internal::requires_execution_supported<E> = 0>
void repeat_until_multi_impl(polymorphic_execution & e, Generator && in,
      pipeline_info<polymorphic_execution,MoreTransformers...> && op, Predicate && predicate_op, Consumer && out)
{
  if (typeid(E) == e.type() && typeid(E) == op.exectype.type()) {
    auto & pipe_exec = op.exectype;
    auto & actual_exec = *pipe_exec. template execution_ptr<E>(); 
    auto transformed_pipe = transform_pipeline(actual_exec, std::forward<std::tuple<MoreTransformers...>>(op.stages)); 
    repeat_until(*e.execution_ptr<E>(),
        std::forward<Generator>(in),
        std::forward<pipeline_info<E,MoreTransformers...> >(transformed_pipe), std::forward<Predicate>(predicate_op),
        std::forward<Consumer>(out));
  }
  else {
    repeat_until_multi_impl<O...>(e, std::forward<Generator>(in),
        std::forward<pipeline_info<polymorphic_execution,MoreTransformers...> >(op), std::forward<Predicate>(predicate_op),
        std::forward<Consumer>(out));
  }
}

template<typename Generator, typename Predicate, typename Consumer,
         typename Operation>
void repeat_until_multi_impl(polymorphic_execution & e, Generator && in, 
      Operation && op, Predicate && predicate_op, Consumer && out)
{
}

template <typename E, typename ... O,
          typename Generator, typename Predicate, typename Consumer,
          typename Operation,
          internal::requires_execution_not_supported<E> = 0>
void repeat_until_multi_impl(polymorphic_execution & e,  Generator && in, 
      Operation && op, Predicate && predicate_op, Consumer && out) 
{
  repeat_until_multi_impl<O...>(e, std::forward<Generator>(in), 
    std::forward<Operation>(op), std::forward<Predicate>(predicate_op), 
    std::forward<Consumer>(out));
}

template <typename E, typename ... O,
          typename Generator, typename Predicate, typename Consumer,
          typename Operation,
          internal::requires_execution_supported<E> = 0>
void repeat_until_multi_impl(polymorphic_execution & e, Generator && in, 
      Operation && op, Predicate && predicate_op, Consumer && out) 
{
  if (typeid(E) == e.type()) {
    repeat_until(*e.execution_ptr<E>(), 
        std::forward<Generator>(in), 
        std::forward<Operation>(op), std::forward<Predicate>(predicate_op), 
        std::forward<Consumer>(out));
  }
  else {
    repeat_until_multi_impl<O...>(e, std::forward<Generator>(in), 
        std::forward<Operation>(op), std::forward<Predicate>(predicate_op), 
        std::forward<Consumer>(out));
  }
}

/**
\addtogroup stream_iteration_pattern
@{
\addtogroup stream_iteration_pattern_poly Polymorphic stream iteration pattern
\brief Sequential implementation of the \ref md_stream-iteration.
@{
*/


/**
\brief Invoke \ref md_stream-iteration on a data stream with polymorphic 
execution with a generator, a predicate, a consumer and a farm as a transformer.
\tparam Generator Callable type for the generation operation.
\tparam Predicate Callable type for the predicate operation.
\tparam Consumer Callable type for the consume operation.
\tparam Transformer Callable type for the transformer operations.
\param ex Polymorphic execution policy object.
\param generate_op Generator operation.
\param farm_obj Composed farm object.
\param predicate_op Predicate operation.
\param consume_op Consumer operation.
*/
template <typename Generator, typename Predicate, typename Consumer,
          typename Transformer>
void repeat_until(polymorphic_execution & ex, 
                  Generator && generate_op,
                  farm_info<polymorphic_execution, Transformer> && farm_obj, 
                  Predicate && predicate_op, Consumer && consume_op)
{
  repeat_until_multi_impl<
    sequential_execution,
    parallel_execution_native
  >(ex, std::forward<Generator>(generate_op), 
      std::forward<farm_info<polymorphic_execution,Transformer> >(farm_obj),
      std::forward<Predicate>(predicate_op), 
      std::forward<Consumer>(consume_op));
}

/**
\brief Invoke \ref md_stream-iteration on a data stream with polymorphic 
execution with a generator, a predicate, a consumer and a pipeline as a transformer.
\tparam Generator Callable type for the generation operation.
\tparam Predicate Callable type for the predicate operation.
\tparam Consumer Callable type for the consume operation.
\tparam MoreTransformers Callable type for the transformer operations.
\param ex Polymorphic execution policy object.
\param generate_op Generator operation.
\param pipe_obj Composed pipeline object.
\param predicate_op Predicate operation.
\param consume_op Consumer operation.
*/
template <typename Generator, typename Predicate, typename Consumer,
          typename ... Transformers>
void repeat_until(polymorphic_execution & ex, Generator && generate_op,
      pipeline_info<polymorphic_execution, Transformers...> && pipe_info, 
      Predicate && predicate_op, Consumer && consume_op)
{
  repeat_until_multi_impl<
    sequential_execution,
    parallel_execution_native
  >(ex, std::forward<Generator>(generate_op), 
      std::forward<pipeline_info<polymorphic_execution,Transformers...>>(pipe_info),
      std::forward<Predicate>(predicate_op), 
      std::forward<Consumer>(consume_op));
}

/**
\brief Invoke \ref md_stream-iteration on a data stream with polymorphic 
execution with a generator, a predicate, a transformer and a consumer.
\tparam Generator Callable type for the generation operation.
\tparam Predicate Callable type for the predicate operation.
\tparam Consumer Callable type for the consume operation.
\tparam Transformer Callable type for the transformer operation.
\param ex Parallel native execution policy object.
\param generate_op Generator operation.
\param tranformer_op Tranformer operation.
\param predicate_op Predicate operation.
\param consume_op Consumer operation.
*/
template <typename Generator, typename Transformer, typename Predicate, typename Consumer>
void repeat_until(polymorphic_execution & ex, 
                  Generator && generate_op, 
                  Transformer && transform_op, Predicate && predicate_op, 
                  Consumer && consume_op) 
{
  repeat_until_multi_impl<
    sequential_execution,
    parallel_execution_native
  >(ex, std::forward<Generator>(generate_op), std::forward<Transformer>(transform_op), 
       std::forward<Predicate>(predicate_op), std::forward<Consumer>(consume_op));
}

} // end namespace grppi

#endif
