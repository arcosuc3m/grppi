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
#ifndef GRPPI_REDUCE_H 
#define GRPPI_REDUCE_H

#include <utility>

#include "common/range_concept.h"
#include "common/iterator_traits.h"
#include "common/execution_traits.h"
#include "common/callable_traits.h"

namespace grppi {

/** 
\addtogroup data_patterns
@{
\defgroup reduce_pattern Reduce pattern
\brief Interface for applyinng the \ref md_reduce.
@{
*/

/**
\brief Invoke \ref md_reduce with identity value
on a data sequence with sequential execution.
\tparam Execution Execution type.
\tparam InRange Range type for the input range.
\tparam Result Type for the identity value.
\tparam Combiner Callable type for the combiner operation.
\param ex Execution policy object.
\param rin Input range.
\param identity Identity value for the combiner operation.
\param combiner_op Combiner operation for the reduction.
\return The result of the reduction.
*/
template <typename Execution, typename InRange, typename Result, typename Combiner,
    meta::requires<range_concept,InRange> = 0,
    meta::requires_integral_value<callable_arity,Combiner,2> = 0>
auto reduce(const Execution & ex,
            InRange && rin,
            Result && identity,
            Combiner && combine_op)
{
  static_assert(supports_reduce<Execution>(),
                "reduce not supported on execution type");
//  static_assert(std::is_same<Result,typename std::result_of<Combiner(Result,Result)>::type>::value,
//                "reduce combiner should be homogeneous:T = op(T,T)");
  return ex.reduce(rin.begin(), rin.size(),
                   std::forward<Result>(identity), std::forward<Combiner>(combine_op));
}


/**
\brief Invoke \ref md_reduce with identity value
on a data sequence with sequential execution.
\tparam Execution Execution type.
\tparam InputIt Iterator type used for input sequence.
\tparam Result Type for the identity value.
\tparam Combiner Callable type for the combiner operation.
\param ex Execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param identity Identity value for the combiner operation.
\param combiner_op Combiner operation for the reduction.
\return The result of the reduction.
*/
template <typename Execution, typename InputIt, typename Result, typename Combiner,
          requires_iterator<InputIt> = 0,
          meta::requires_integral_value<callable_arity,Combiner,2> = 0>
auto reduce(const Execution & ex, 
            InputIt first, InputIt last, 
            Result && identity,
            Combiner && combine_op)
{
  static_assert(supports_reduce<Execution>(),
      "reduce not supported on execution type");
//  static_assert(std::is_same<Result,typename std::result_of<Combiner(Result,Result)>::type>::value,
//                "reduce combiner should be homogeneous:T = op(T,T)");
  return ex.reduce(first, std::distance(first,last), 
      std::forward<Result>(identity), std::forward<Combiner>(combine_op));
}

/**
\brief Invoke \ref md_reduce with identity value
on a data sequence with sequential execution.
\tparam Execution Execution type.
\tparam InputIt Iterator type used for input sequence.
\tparam Result Type for the identity value.
\tparam Combiner Callable type for the combiner operation.
\param ex Execution policy object.
\param first Iterator to the first element in the input sequence.
\param size Size of the input sequence to be process.
\param identity Identity value for the combiner operation.
\param combiner_op Combiner operation for the reduction.
\return The result of the reduction.
*/
template <typename Execution, typename InputIt, typename Result, typename Combiner,
    requires_iterator<InputIt> = 0,
    meta::requires_integral_value<callable_arity,Combiner,2> = 0>
auto reduce(const Execution & ex,
            InputIt first, std::size_t size,
            Result && identity,
            Combiner && combine_op)
{
  static_assert(supports_reduce<Execution>(),
                "reduce not supported on execution type");
//  static_assert(std::is_same<Result,typename std::result_of<Combiner(Result,Result)>::type>::value,
//                "reduce combiner should be homogeneous:T = op(T,T)");
  return ex.reduce(first, size,
                   std::forward<Result>(identity), std::forward<Combiner>(combine_op));
}

/**
@}
@}
*/
}

#endif
