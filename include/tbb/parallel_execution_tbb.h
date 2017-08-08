/**
* @version		GrPPI v0.3
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

#ifndef GRPPI_TBB_PARALLEL_EXECUTION_TBB_H
#define GRPPI_TBB_PARALLEL_EXECUTION_TBB_H

#ifdef GRPPI_TBB

#include "../common/mpmc_queue.h"
#include "../common/iterator.h"

#include <type_traits>
#include <tuple>

#include <tbb/tbb.h>

namespace grppi {

/** 
 \brief TBB parallel execution policy.

 This policy uses Intel Threading Building Blocks as implementation back end.
 */
class parallel_execution_tbb {
public:

  /** 
  \brief Default construct a TBB parallel execution policy.

  Creates a TBB parallel execution object.

  The concurrency degree is determined by the platform.

  */
  parallel_execution_tbb() noexcept :
      parallel_execution_tbb{default_concurrency_degree}
  {}

  /** 
  \brief Constructs a TBB parallel execution policy.

  Creates a TBB parallel execution object selecting the concurrency degree.

  \param concurrency_degree Number of threads used for parallel algorithms.
  \param order Whether ordered executions is enabled or disabled.
  */
  parallel_execution_tbb(int concurrency_degree, bool order = true) noexcept :
      concurrency_degree_{concurrency_degree},
      ordering_{order}
  {}

  /**
  \brief Set number of grppi threads.
  */
  void set_concurrency_degree(int degree) noexcept { concurrency_degree_ = degree; }

  /**
  \brief Get number of grppi trheads.
  */
  int concurrency_degree() const noexcept { return concurrency_degree_; }

  /**
  \brief Enable ordering.
  */
  void enable_ordering() noexcept { ordering_=true; }

  /**
  \brief Disable ordering.
  */
  void disable_ordering() noexcept { ordering_=false; }

  /**
  \brief Is execution ordered.
  */
  bool is_ordered() const noexcept { return ordering_; }

  /**
  \brief Sets the attributes for the queues built through make_queue<T>()
  */
  void set_queue_attributes(int size, queue_mode mode, int tokens) noexcept {
    queue_size_ = size;
    queue_mode_ = mode;
    num_tokens_ = tokens;
  }

  /**
  \brief Makes a communication queue for elements of type T.
  Constructs a queue using the attributes that can be set via 
  set_queue_attributes(). The value is returned via move semantics.
  */
  template <typename T>
  mpmc_queue<T> make_queue() const {
    return {queue_size_, queue_mode_};
  }

  /**
  \brien Get num of tokens.
  */
  int tokens() const noexcept { return num_tokens_; }

  /**
  \brief Applies a trasnformation to multiple sequences leaving the result in
  another sequence using available TBB parallelism.
  \tparam InputIterators Iterator types for input sequences.
  \tparam OutputIterator Iterator type for the output sequence.
  \tparam Transformer Callable object type for the transformation.
  \param firsts Tuple of iterators to input sequences.
  \param first_out Iterator to the output sequence.
  \param sequence_size Size of the input sequences.
  \param transform_op Transformation callable object.
  \pre For every I iterators in the range 
       `[get<I>(firsts), next(get<I>(firsts),sequence_size))` are valid.
  \pre Iterators in the range `[first_out, next(first_out,sequence_size)]` are valid.
  */
  template <typename ... InputIterators, typename OutputIterator, 
            typename Transformer>
  void apply_map(std::tuple<InputIterators...> firsts,
      OutputIterator first_out, 
      std::size_t sequence_size, Transformer transform_op) const;

private:

  constexpr static int default_concurrency_degree = 4;
  int concurrency_degree_ = default_concurrency_degree;

  bool ordering_ = true;

  constexpr static int default_queue_size = 100;
  int queue_size_ = default_queue_size;

  constexpr static int default_num_tokens_ = 100;
  int num_tokens_ = default_num_tokens_;

  queue_mode queue_mode_ = queue_mode::blocking;
};

template <typename ... InputIterators, typename OutputIterator, 
          typename Transformer>
void parallel_execution_tbb::apply_map(
    std::tuple<InputIterators...> firsts,
    OutputIterator first_out, 
    std::size_t sequence_size, Transformer transform_op) const
{
  tbb::parallel_for(
    std::size_t{0}, sequence_size, 
    [&] (std::size_t index){
      first_out[index] = apply_iterators_indexed(transform_op, firsts, index);
    }
 );   

}

/**
\brief Metafunction that determines if type E is parallel_execution_tbb
\tparam Execution policy type.
*/
template <typename E>
constexpr bool is_parallel_execution_tbb() {
  return std::is_same<E, parallel_execution_tbb>::value;
}

/**
\brief Metafunction that determines if type E is supported in the current build.
\tparam Execution policy type.
*/
template <typename E>
constexpr bool is_supported();

/**
\brief Specialization stating that parallel_execution_tbb is supported.
This metafunction evaluates to false if GRPPI_TBB is enabled.
*/
template <>
constexpr bool is_supported<parallel_execution_tbb>() {
  return true;
}


} // end namespace grppi

#else // GRPPI_TBB not defined

namespace grppi {


/// Parallel execution policy.
/// Empty type if GRPPI_TBB disabled.
struct parallel_execution_tbb {};

/**
\brief Metafunction that determines if type E is parallel_execution_tbb
This metafunction evaluates to false if GRPPI_TBB is disabled.
\tparam Execution policy type.
*/
template <typename E>
constexpr bool is_parallel_execution_tbb() {
  return false;
}

/**
\brief Metafunction that determines if type E is supported in the current build.
\tparam Execution policy type.
*/
template <typename E>
constexpr bool is_supported();

/**
\brief Specialization stating that parallel_execution_native is supported.
*/
template <>
constexpr bool is_supported<parallel_execution_tbb>() {
  return false;
}

} // end namespace grppi

#endif // GRPPI_TBB

#endif
