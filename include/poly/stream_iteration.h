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

template<typename GenFunc, typename Predicate, typename OutFunc,
         typename Transformer >
void repeat_until_multi_impl(polymorphic_execution & e, GenFunc && in,
      farm_info<polymorphic_execution,Transformer> && op, Predicate && condition, OutFunc && out)
{
}

template <typename E, typename ... O,
          typename GenFunc, typename Predicate, typename OutFunc,
          typename Transformer ,
          internal::requires_execution_not_supported<E> = 0>
void repeat_until_multi_impl(polymorphic_execution & e,  GenFunc && in,
      pipeline_info<polymorphic_execution,Transformer> && op, Predicate && condition, OutFunc && out)
{
  repeat_until_multi_impl<O...>(e, std::forward<GenFunc>(in),
    std::forward<farm_info<polymorphic_execution,Transformer> >(op), std::forward<Predicate>(condition),
    std::forward<OutFunc>(out));
}

template <class E, typename ... O,
          typename GenFunc, typename Predicate, typename OutFunc,
          typename Transformer ,
          internal::requires_execution_supported<E> = 0>
void repeat_until_multi_impl(polymorphic_execution & e, GenFunc && in,
      farm_info<polymorphic_execution,Transformer> && op, Predicate && condition, OutFunc && out)
{
  if (typeid(E) == e.type() && typeid(E) == op.exectype.type()) {
    auto & pipe_exec = op.exectype;
    auto & actual_exec = *pipe_exec. template execution_ptr<E>();
    auto transformed_farm = farm(actual_exec, std::forward<Transformer>(op.task));
    repeat_until(*e.execution_ptr<E>(),
        std::forward<GenFunc>(in),
        std::forward<farm_info<E,Transformer> >(transformed_farm), std::forward<Predicate>(condition),
        std::forward<OutFunc>(out));
  }
  else {
    repeat_until_multi_impl<O...>(e, std::forward<GenFunc>(in),
        std::forward<farm_info<polymorphic_execution,Transformer> >(op), std::forward<Predicate>(condition),
        std::forward<OutFunc>(out));
  }
}

template<typename GenFunc, typename Predicate, typename OutFunc,
         typename ...MoreTransformers >
void repeat_until_multi_impl(polymorphic_execution & e, GenFunc && in,
      pipeline_info<polymorphic_execution,MoreTransformers...> && op, Predicate && condition, OutFunc && out)
{
}

template <typename E, typename ... O,
          typename GenFunc, typename Predicate, typename OutFunc,
          typename ...MoreTransformers ,
          internal::requires_execution_not_supported<E> = 0>
void repeat_until_multi_impl(polymorphic_execution & e,  GenFunc && in,
      pipeline_info<polymorphic_execution,MoreTransformers...> && op, Predicate && condition, OutFunc && out)
{
  repeat_until_multi_impl<O...>(e, std::forward<GenFunc>(in),
    std::forward<pipeline_info<polymorphic_execution,MoreTransformers...> >(op), std::forward<Predicate>(condition),
    std::forward<OutFunc>(out));
}

template <class E, typename ... O,
          typename GenFunc, typename Predicate, typename OutFunc,
          typename ...MoreTransformers ,
          internal::requires_execution_supported<E> = 0>
void repeat_until_multi_impl(polymorphic_execution & e, GenFunc && in,
      pipeline_info<polymorphic_execution,MoreTransformers...> && op, Predicate && condition, OutFunc && out)
{
  if (typeid(E) == e.type() && typeid(E) == op.exectype.type()) {
    auto & pipe_exec = op.exectype;
    auto & actual_exec = *pipe_exec. template execution_ptr<E>(); 
    auto transformed_pipe = transform_pipeline(actual_exec, std::forward<std::tuple<MoreTransformers...>>(op.stages)); 
    repeat_until(*e.execution_ptr<E>(),
        std::forward<GenFunc>(in),
        std::forward<pipeline_info<E,MoreTransformers...> >(transformed_pipe), std::forward<Predicate>(condition),
        std::forward<OutFunc>(out));
  }
  else {
    repeat_until_multi_impl<O...>(e, std::forward<GenFunc>(in),
        std::forward<pipeline_info<polymorphic_execution,MoreTransformers...> >(op), std::forward<Predicate>(condition),
        std::forward<OutFunc>(out));
  }
}

template<typename GenFunc, typename Predicate, typename OutFunc,
         typename Operation>
void repeat_until_multi_impl(polymorphic_execution & e, GenFunc && in, 
      Operation && op, Predicate && condition, OutFunc && out)
{
}

template <typename E, typename ... O,
          typename GenFunc, typename Predicate, typename OutFunc,
          typename Operation,
          internal::requires_execution_not_supported<E> = 0>
void repeat_until_multi_impl(polymorphic_execution & e,  GenFunc && in, 
      Operation && op, Predicate && condition, OutFunc && out) 
{
  repeat_until_multi_impl<O...>(e, std::forward<GenFunc>(in), 
    std::forward<Operation>(op), std::forward<Predicate>(condition), 
    std::forward<OutFunc>(out));
}

template <typename E, typename ... O,
          typename GenFunc, typename Predicate, typename OutFunc,
          typename Operation,
          internal::requires_execution_supported<E> = 0>
void repeat_until_multi_impl(polymorphic_execution & e, GenFunc && in, 
      Operation && op, Predicate && condition, OutFunc && out) 
{
  if (typeid(E) == e.type()) {
    repeat_until(*e.execution_ptr<E>(), 
        std::forward<GenFunc>(in), 
        std::forward<Operation>(op), std::forward<Predicate>(condition), 
        std::forward<OutFunc>(out));
  }
  else {
    repeat_until_multi_impl<O...>(e, std::forward<GenFunc>(in), 
        std::forward<Operation>(op), std::forward<Predicate>(condition), 
        std::forward<OutFunc>(out));
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
\param predicate_op Predicate operation.
\param consume_op Consumer operation.
\param farm Composed farm object.
*/
template <typename GenFunc, typename Predicate, typename OutFunc,
          typename Transformer>
void repeat_until(polymorphic_execution & e, GenFunc && in,
      farm_info<polymorphic_execution, Transformer> && op, Predicate && condition, OutFunc && out)
{
  repeat_until_multi_impl<
    sequential_execution,
    parallel_execution_native
  >(e, std::forward<GenFunc>(in), std::forward<farm_info<polymorphic_execution,Transformer> >(op),
       std::forward<Predicate>(condition), std::forward<OutFunc>(out));
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
\param predicate_op Predicate operation.
\param consume_op Consumer operation.
\param pipe Composed pipeline object.
*/
template <typename GenFunc, typename Predicate, typename OutFunc,
          typename ...MoreTransformers>
void repeat_until(polymorphic_execution & e, GenFunc && in,
      pipeline_info<polymorphic_execution, MoreTransformers...> && op, Predicate && condition, OutFunc && out)
{
  repeat_until_multi_impl<
    sequential_execution,
    parallel_execution_native
  >(e, std::forward<GenFunc>(in), std::forward<pipeline_info<polymorphic_execution,MoreTransformers...> >(op),
       std::forward<Predicate>(condition), std::forward<OutFunc>(out));
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
\param predicate_op Predicate operation.
\param consume_op Consumer operation.
\param tranformer_op Tranformer operation.
*/
template <typename GenFunc, typename Operation, typename Predicate, typename OutFunc>
void repeat_until(polymorphic_execution & e, GenFunc && in, 
      Operation && op, Predicate && condition, OutFunc && out) 
{
  repeat_until_multi_impl<
    sequential_execution,
    parallel_execution_native
  >(e, std::forward<GenFunc>(in), std::forward<Operation>(op), 
       std::forward<Predicate>(condition), std::forward<OutFunc>(out));
}

} // end namespace grppi

#endif
