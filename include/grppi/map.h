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
#ifndef GRPPI_MAP_H
#define GRPPI_MAP_H

#include <utility>
#include <tuple>

#include "grppi/common/zip_view.h"
#include "grppi/common/execution_traits.h"
#include "grppi/common/iterator_traits.h"

namespace grppi {

/** 
\addtogroup data_patterns
@{
\defgroup map_pattern Map pattern
\brief Interface for applying the \ref md_map.
@{
*/

/**
\brief Invoke \ref md_map on a data sequence.
\tparam Execution Execution policy type.
\tparam InRange Range type for the input range.
\tparam OutRange Range type for the output range.
\tparam Transformer Callable type for the transformation operation.
\param ex Execution policy object.
\param rin Input range.
\param rout Output range.
\param transform_op Transformation operation.
\pre rin.size() == rout.size()
*/
  template<typename Execution, typename InRange, typename OutRange,
      typename Transformer,
      meta::requires_<range_concept, InRange> = 0,
      meta::requires_<range_concept, OutRange> = 0>

  void map(const Execution & ex, InRange && rin, OutRange && rout,
      Transformer transform_op)
  {
    static_assert(supports_map<Execution>(),
        "map not supported on execution type");
    ex.map(std::make_tuple(rin.begin()), rout.begin(),
        rin.size(), transform_op);
  }

  /**
\brief Invoke \ref md_map on arrays.
\tparam Execution Execution policy type.
\tparam T Element type for input array.
\tparam U Element type for output array.
\tparam N Size of input and output array.
\tparam Transformer Callable type for the transformation operation.
\param ex Execution policy object.
\param array_in Input array.
\param array_out Output array.
\param transform_op Transformation operation.
*/
  template<typename Execution, typename T, typename U,
      std::size_t N,
      typename Transformer>
  void map(const Execution & ex, T (& array_in)[N], U (& array_out)[N],
      Transformer transform_op)
  {
    static_assert(supports_map<Execution>(),
        "map not supported on execution type");
    ex.map(std::make_tuple(std::begin(array_in)), std::begin(array_out),
        N, transform_op);
  }

/**
\brief Invoke \ref md_map on a data sequence.
\tparam Execution Execution policy type.
\tparam InTs Element types for the input arrays.
\tparam OutTs Element type for the output arrays.
\tparam Transformer Callable type for the transformation operation.
\param ex Execution policy object.
\param a_ins Input arrays packaged in a zip_view_arrays.
\param a_out Output array.
\param transform_op Transformation operation.
*/
  template<typename Execution, typename ... InTs, typename OutT,
      std::size_t N,
      typename Transformer>
  void map(const Execution & ex, zip_view_arrays<N, InTs...> a_ins,
      OutT (& a_out)[N],
      Transformer transform_op)
  {
    static_assert(supports_map<Execution>(),
        "map not supported on execution type");
    ex.map(a_ins.begin(), a_out, N, transform_op);
  }

/**
\brief Invoke \ref md_map on a data sequence.
\tparam Execution Execution policy type.
\tparam InRanges Range types for the input ranges.
\tparam OutRange Range type for the output range.
\tparam Transformer Callable type for the transformation operation.
\param ex Execution policy object.
\param range_ins Input ranges packaged in a std::tuple.
\param range_out Output range.
\param transform_op Transformation operation.
\pre for all r in range_ins: r.size() == range_out.size()
*/
  template<typename Execution, typename ... InRanges, typename OutRange,
      typename Transformer,
      meta::requires_<range_concept, InRanges...> = 0,
      meta:: requires_<range_concept, OutRange> = 0>

  void map(const Execution & ex, zip_view<InRanges...> range_ins,
      OutRange && range_out,
      Transformer transform_op)
  {
    static_assert(supports_map<Execution>(),
        "map not supported on execution type");
    ex.map(range_ins.begin(), range_out.begin(),
        range_out.size(), transform_op);
  }

/**
\brief Invoke \ref md_map on a data sequence.
\tparam InputIt Iterator type used for input sequence.
\tparam OutputIt Iterator type used for the output sequence.
\tparam Transformer Callable type for the transformation operation.
\param ex Execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param first_out Iterator to first element of the output sequence.
\param transform_op Transformation operation.
*/
  template<typename Execution, typename InputIt, typename OutputIt,
      typename Transformer,
      requires_iterator<InputIt> = 0,
      requires_iterator<OutputIt> = 0>
  void map(const Execution & ex,
      InputIt first, InputIt last, OutputIt first_out,
      Transformer transform_op)
  {
    static_assert(supports_map<Execution>(),
        "map not supported on execution type");
    ex.map(std::make_tuple(first), first_out,
        std::distance(first, last), transform_op);
  }

/**
\brief Invoke \ref md_map on a data sequence.
\tparam InputIterators Iterator types used for input sequences.
\tparam InputIt Iterator type used for any input sequence.
\tparam OutputIt Iterator type used for the output sequence.
\tparam Transformer Callable type for the transformation operation.
\param ex Execution policy object.
\param firsts Tuple of Iterators to the first element in the inputs sequences.
\param last Iterator to one past the end of one of the input sequence.
\param first_out Iterator to first element of the output sequence.
\param transform_op Transformation operation.
*/
  template<typename Execution, typename ...InputIterators, typename InputIt,
      typename OutputIt, typename Transformer,
      requires_iterators<InputIterators...> = 0,
      requires_iterator<InputIt> = 0,
      requires_iterator<OutputIt> = 0>
  void map(const Execution & ex, std::tuple<InputIterators...> firsts,
      InputIt last, OutputIt first_out,
      Transformer transform_op)
  {
    static_assert(supports_map<Execution>(),
        "map not supported on execution type");
    ex.map(firsts, first_out,
        std::distance(std::get<0>(firsts), last), transform_op);
  }

/**
\brief Invoke \ref md_map on a data sequence.
\tparam InputIt Iterator type used for input sequence.
\tparam OutputIt Iterator type used for the output sequence.
\tparam Transformer Callable type for the transformation operation.
\param ex Execution policy object.
\param first Iterator to the first element in the input sequence.
\param size Size of the input sequence.
\param first_out Iterator to first element of the output sequence.
\param transform_op Transformation operation.
*/
  template<typename Execution, typename InputIt, typename OutputIt,
      typename Transformer,
      requires_iterator<InputIt> = 0,
      requires_iterator<OutputIt> = 0>
  void map(const Execution & ex,
      InputIt first, std::size_t size, OutputIt first_out,
      Transformer transform_op)
  {
    static_assert(supports_map<Execution>(),
        "map not supported on execution type");
    ex.map(std::make_tuple(first), first_out, size, transform_op);
  }


/**
\brief Invoke \ref md_map on a data sequence.
\tparam InputIterators Iterator types used for input sequences.
\tparam OutputIt Iterator type used for the output sequence.
\tparam Transformer Callable type for the transformation operation.
\param ex Execution policy object.
\param firsts Tuple of Iterators to the first element in the inputs sequences.
\param size Size of the input sequences.
\param first_out Iterator to first element of the output sequence.
\param transform_op Transformation operation.
*/
  template<typename Execution, typename ...InputIterators,
      typename OutputIt, typename Transformer,
      requires_iterators<InputIterators...> = 0,
      requires_iterator<OutputIt> = 0>
  void map(const Execution & ex, std::tuple<InputIterators...> firsts,
      std::size_t size, OutputIt first_out,
      Transformer transformer_op)
  {
    static_assert(supports_map<Execution>(),
        "map not supported on execution type");
    ex.map(firsts, first_out, size, transformer_op);
  }

/**
\brief Invoke \ref md_map on a data sequence.
\tparam InputIt Iterator type used for input sequence.
\tparam OutputIt Iterator type used for the output sequence.
\tparam Transformer Callable type for the transformation operation.
\tparam OtherInputIts Iterator types used for additional input sequences.
\param ex Execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param first_out Iterator to first element of the output sequence.
\param transform_op Transformation operation.
\param more_firsts Additional iterators with first elements of additional sequences.
\deprecated For multiple inputs, use tuple or zip versions.
*/
  template<typename Execution, typename InputIt, typename OutputIt, typename Transformer,
      typename ... OtherInputIts,
      requires_iterator<InputIt> = 0,
      requires_iterator<OutputIt> = 0>
  [[deprecated("This version of the interface is deprecated.\n"
  "For multiple inputs, use a tuple or zip versions.")]]
  void map(const Execution & ex,
      InputIt first, InputIt last, OutputIt first_out,
      Transformer transform_op,
      OtherInputIts ... other_firsts)
  {
    static_assert(supports_map<Execution>(),
        "map not supported on execution type");
    ex.map(std::make_tuple(first, other_firsts...), first_out,
        std::distance(first, last), transform_op);
  }

/**
@}
@}
*/
}

#endif
