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

#ifndef GRPPI_TBB_PARALLEL_EXECUTION_TBB_H
#define GRPPI_TBB_PARALLEL_EXECUTION_TBB_H

// Only if compiled with TBB enabled
#ifdef GRPPI_TBB

#include <type_traits>

#include "../common/mpmc_queue.h"

namespace grppi{

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
