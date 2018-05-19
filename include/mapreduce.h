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
#ifndef GRPPI_MAPREDUCE_H 
#define GRPPI_MAPREDUCE_H

#include <utility>

#include "range_mapping.h"
#include "common/execution_traits.h"
#include "common/iterator_traits.h"

namespace grppi {

/**
\addtogroup data_patterns
@{
\defgroup mapreduce_pattern Map/reduce pattern
\brief Interface for applyinng the \ref md_map-reduce.
@{
*/

/**
\brief Invoke \ref md_map-reduce on a data sequence.
\tparam Execution Execution type.
\tparam InputIterator Iterator type used for the input sequence.
\tparam Identity Type for the identity value.
\tparam Transformer Callable type for the transformation operation.
\tparam Combiner Callable type for the combination operation of the reduction.
\param ex Execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param identity Identity value for the combination operation.
\param transf_op Transformation operation.
\param combine_op Combination operation.
\return Result of the map/reduce operation.
*/
template <typename Execution, typename InputRange, typename Identity, 
          typename Transformer, typename Combiner,
          meta::requires<range_concept,InputRange> = 0>
auto map_reduce(const Execution & ex, 
                InputRange && rin,
                Identity && identity, 
                Transformer &&  transform_op, Combiner && combine_op)
{
  static_assert(supports_map_reduce<Execution>(),
    "map/reduce not supported on execution type");
  return ex.map_reduce(std::make_tuple(rin.begin()), rin.size(),
      std::forward<Identity>(identity),
      std::forward<Transformer>(transform_op), 
      std::forward<Combiner>(combine_op));
}

/**
\brief Invoke \ref md_map-reduce on a data sequence.
\tparam Execution Execution type.
\tparam InputIterators Iterators types used for the input sequences.
\tparam Identity Type for the identity value.
\tparam Transformer Callable type for the transformation operation.
\tparam Combiner Callable type for the combination operation of the reduction.
\param ex Execution policy object.
\param firsts Tuple of iterators to the first elements in the input sequences.
\param size Size of the input sequence to be process.
\param identity Identity value for the combination operation.
\param transf_op Transformation operation.
\param combine_op Combination operation.
\return Result of the map/reduce operation.
*/
template <typename Execution, 
    typename ... InputRanges,
    typename Identity, typename Transformer, typename Combiner,
    meta::requires<range_concept,InputRanges ...> = 0>
auto map_reduce(const Execution & ex,
                zip_view<InputRanges...> rins,
                Identity && identity,
                Transformer &&  transform_op, Combiner && combine_op)
{
  static_assert(supports_map_reduce<Execution>(),
                "map/reduce not supported on execution type");
  return ex.map_reduce(rins.begin(), rins.size(),
                       std::forward<Identity>(identity),
                       std::forward<Transformer>(transform_op),
                       std::forward<Combiner>(combine_op));
}

/**
\brief Invoke \ref md_map-reduce on a data sequence.
\tparam Execution Execution type.
\tparam InputIterators Iterators types used for the input sequences.
\tparam Identity Type for the identity value.
\tparam Transformer Callable type for the transformation operation.
\tparam Combiner Callable type for the combination operation of the reduction.
\param ex Execution policy object.
\param firsts Tuple of iterators to the first elements in the input sequences.
\param size Size of the input sequence to be process.
\param identity Identity value for the combination operation.
\param transf_op Transformation operation.
\param combine_op Combination operation.
\return Result of the map/reduce operation.
*/
template <typename Execution, typename ...InputIterators,
    typename Identity, typename Transformer, typename Combiner,
    requires_iterators<InputIterators...> = 0>
auto map_reduce(const Execution & ex,
                std::tuple<InputIterators...> firsts, std::size_t size,
                Identity && identity,
                Transformer &&  transform_op, Combiner && combine_op)
{
  static_assert(supports_map_reduce<Execution>(),
                "map/reduce not supported on execution type");
  return ex.map_reduce(firsts, size,
                       std::forward<Identity>(identity),
                       std::forward<Transformer>(transform_op),
                       std::forward<Combiner>(combine_op));
}

/**
\brief Invoke \ref md_map-reduce on a data sequence.
\tparam Execution Execution type.
\tparam InputIterators Iterators types used for the input sequences.
\tparam InputIt Iterator type used for the fisrt input sequence.
\tparam Identity Type for the identity value.
\tparam Transformer Callable type for the transformation operation.
\tparam Combiner Callable type for the combination operation of the reduction.
\param ex Execution policy object.
\param firsts Tuple of iterators to the first elements in the input sequences.
\param last Iterator to one past the end of the input sequence.
\param identity Identity value for the combination operation.
\param transf_op Transformation operation.
\param combine_op Combination operation.
\return Result of the map/reduce operation.
*/
template <typename Execution, typename ...InputIterators, typename InputIt,
          typename Identity, typename Transformer, typename Combiner,
          requires_iterators<InputIterators...> = 0,
          requires_iterator<InputIt> = 0>
auto map_reduce(const Execution & ex,
                std::tuple<InputIterators...> firsts, InputIt last,
                Identity && identity,
                Transformer &&  transform_op, Combiner && combine_op)
{
  static_assert(supports_map_reduce<Execution>(),
                "map/reduce not supported on execution type");
  return ex.map_reduce(firsts,
                       std::distance(std::get<0>(firsts),last),
                       std::forward<Identity>(identity),
                       std::forward<Transformer>(transform_op),
                       std::forward<Combiner>(combine_op));
}

/**
\brief Invoke \ref md_map-reduce on a data sequence.
\tparam Execution Execution type.
\tparam InputIterator Iterator type used for the input sequence.
\tparam Identity Type for the identity value.
\tparam Transformer Callable type for the transformation operation.
\tparam Combiner Callable type for the combination operation of the reduction.
\param ex Execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param identity Identity value for the combination operation.
\param transf_op Transformation operation.
\param combine_op Combination operation.
\return Result of the map/reduce operation.
*/
template <typename Execution, typename InputIterator, typename Identity, 
          typename Transformer, typename Combiner,
          requires_iterator<InputIterator> = 0>
auto map_reduce(const Execution & ex, 
                InputIterator first, InputIterator last, 
                Identity && identity, 
                Transformer &&  transform_op, Combiner && combine_op)
{
  static_assert(supports_map_reduce<Execution>(),
    "map/reduce not supported on execution type");
  return ex.map_reduce(std::make_tuple(first), std::distance(first,last), 
      std::forward<Identity>(identity),
      std::forward<Transformer>(transform_op), 
      std::forward<Combiner>(combine_op));
}

/**
\brief Invoke \ref md_map-reduce on multiple data sequences. 
\tparam Execution Execution type.
\tparam InputIterator Iterator type used for the input sequence.
\tparam Identity Type for the identity value.
\tparam Transformer Callable type for the transformation operation.
\tparam Combiner Callable type for the combination operation of the reduction.
\param ex Execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param identity Identity value for the combination operation.
\param transf_op Transformation operation.
\param combine_op Combination operation.
\return Result of the map/reduce operation.
*/
template <typename Execution, typename InputIterator, typename Identity, 
          typename Transformer, typename Combiner,
          typename ... OtherInputIterators,
          requires_iterator<InputIterator> = 0>
[[deprecated("This version of the interface is deprecated.\n"
             "If you want to use multiple inputs, use a tuple instead.")]]
auto map_reduce(const Execution & ex,
                InputIterator first, InputIterator last, 
                Identity && identity, 
                Transformer &&  transform_op, Combiner && combine_op,
                OtherInputIterators ... other_firsts)
{
  static_assert(supports_map_reduce<Execution>(),
    "map/reduce not supported on execution type");
  return ex.map_reduce(std::make_tuple(first, other_firsts...), 
      std::distance(first,last), 
      std::forward<Identity>(identity),
      std::forward<Transformer>(transform_op), 
      std::forward<Combiner>(combine_op));
}

/**
@}
@}
*/

}

#endif
