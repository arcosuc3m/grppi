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

#ifndef GRPPI_OMP_PARALLEL_EXECUTION_OMP_H
#define GRPPI_OMP_PARALLEL_EXECUTION_OMP_H

#ifdef GRPPI_OMP

#include "../common/mpmc_queue.h"

#include <type_traits>

#include <omp.h>


namespace grppi {

/**
\brief OpenMP parallel execution policy.

This policy uses OpenMP as implementation back-end.
*/
class parallel_execution_omp {

public:
  /** 
  \brief Default construct an OpenMP parallel execution policy.

  Creates an OpenMP parallel execution object.

  The concurrency degree is determined by the platform according to OpenMP 
  rules.
  */
  parallel_execution_omp() noexcept :
      parallel_execution_omp{impl_concurrency_degree()}
  {}

  /** @brief Set num_threads to _threads in order to run in parallel
   *
   *  @param _threads number of threads used in the parallel mode
   */
  /** 
  \brief Constructs an OpenMP parallel execution policy.

  Creates an OpenMP parallel execution object selecting the concurrency degree
  and ordering.
  \param concurrency_degree Number of threads used for parallel algorithms.
  \param order Whether ordered executions is enabled or disabled.
  */
  parallel_execution_omp(int concurrency_degree, bool order = true) noexcept :
      concurrency_degree_{concurrency_degree},
      ordering_{order}
  {
    omp_set_num_threads(concurrency_degree_);
  }

  /**
  \brief Set number of grppi threads.
  */
  void set_concurrency_degree(int degree) noexcept { 
    concurrency_degree_ = degree; 
    omp_set_num_threads(concurrency_degree_);
  }

  /**
  \brief Get number of grppi trheads.
  */
  int concurrency_degree() const noexcept { 
    return concurrency_degree_; 
  }

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
  \brief Sets the attributes for the queues built through make_queue<T>(()
  */
  void set_queue_attributes(int size, queue_mode mode) noexcept {
    queue_size_ = size;
    queue_mode_ = mode;
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
  \brief Get index of current thread in the thread table
  */
  int get_thread_id() const noexcept {
    int result;
    #pragma omp parallel
    {
      result = omp_get_thread_num();
    }
    return result;
  }

private:

  /**
  \brief Obtain OpenMP platform number of threads.
  Queries the current OpenMP number of threads so that it can be used in
  intialization of data members.
  \return The current OpenMP number of threads.
  \note The determination is performed inside a parallel region.
  */
  static int impl_concurrency_degree() {
    int result;
    #pragma omp parallel
    {
      result = omp_get_num_threads();
    }
    return result;
  }

private:

  int concurrency_degree_;

  bool ordering_;

  constexpr static int default_queue_size = 100;
  int queue_size_ = default_queue_size;

  queue_mode queue_mode_ = queue_mode::blocking;
};

/**
\brief Metafunction that determines if type E is parallel_execution_omp
\tparam Execution policy type.
*/
template <typename E>
constexpr bool is_parallel_execution_omp() {
  return std::is_same<E, parallel_execution_omp>::value;
}

/**
\brief Metafunction that determines if type E is supported in the current build.
\tparam Execution policy type.
*/
template <typename E>
constexpr bool is_supported();

/**
\brief Specialization stating that parallel_execution_omp is supported.
This metafunction evaluates to false if GRPPI_OMP is enabled.
*/
template <>
constexpr bool is_supported<parallel_execution_omp>() {
  return true;
}

} // end namespace grppi

#else // GRPPI_OMP undefined

namespace grppi {


/// Parallel execution policy.
/// Empty type if GRPPI_OMP disabled.
struct parallel_execution_omp {};

/**
\brief Metafunction that determines if type E is parallel_execution_omp
This metafunction evaluates to false if GRPPI_OMP is disabled.
\tparam Execution policy type.
*/
template <typename E>
constexpr bool is_parallel_execution_omp() {
  return false;
}

/**
\brief Metafunction that determines if type E is supported in the current build.
\tparam Execution policy type.
*/
template <typename E>
constexpr bool is_supported();

/**
\brief Specialization stating that parallel_execution_omp is supported.
*/
template <>
constexpr bool is_supported<parallel_execution_omp>() {
  return false;
}

}

#endif // GRPPI_OMP

#endif
