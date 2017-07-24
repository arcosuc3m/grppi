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

#ifndef GRPPI_NATIVE_PARALLEL_EXECUTION_NATIVE_H
#define GRPPI_NATIVE_PARALLEL_EXECUTION_NATIVE_H

#include <thread>
#include <atomic>
#include <algorithm>
#include <vector>
#include <type_traits>

#include "pool.h"
#include "common/mpmc_queue.h"

namespace grppi {

/**
\brief Thread index table to provide portable natural thread indices.

A thread table provides a simple way to offer thread indices (starting from 0).

When a thread registers itself in the registry. To get an integer index, users
may call current_index, which provides the order number of the calling thread in
the registry.

\note This class is thread safe by means of using a spin-lock.
*/
class thread_registry {
public:
  thread_registry() noexcept = default;

  /**
  \brief Adds current thread in the registry.
  */
  void register_thread() noexcept;

  /**
  \brief Removes current thread from the registry.
  */
  void deregister_thread() noexcept;

  /**
  \brief Integer index for current thread
  \return Integer value with the registration order of current thread.
  \pre Current thread is registered.
  */
  int current_index() const noexcept;

private:
  mutable std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
  std::vector<std::thread::id> ids_;
};

inline void thread_registry::register_thread() noexcept 
{
  using namespace std;
  while (lock_.test_and_set(memory_order_acquire)) {}
  auto this_id = this_thread::get_id();
  ids_.push_back(this_id);
  lock_.clear(memory_order_release);
}

inline void thread_registry::deregister_thread() noexcept
{
  using namespace std;
  while (lock_.test_and_set(memory_order_acquire)) {}
  auto this_id = this_thread::get_id();
  auto current = find(begin(ids_), end(ids_), this_id);
  *current = {}; //Empty thread
  lock_.clear(memory_order_release);
}

inline int thread_registry::current_index() const noexcept
{
  using namespace std;
  while (lock_.test_and_set(memory_order_acquire)) {}
  auto this_id = this_thread::get_id();
  auto current = find(begin(ids_), end(ids_), this_id);
  auto index = distance(begin(ids_), current);
  lock_.clear(memory_order_release);
  return index;
};

/**
\brief RAII class to mange registration/deregistration pairs.
This class allows to manage automatic deregistration of threads through
the common RAII pattern. The current thread is registered into the registry
at construction and deregistered a destruction.
*/
class native_thread_manager {
public:
  /**
  \brief Saves a reference to the registry and registers current thread
  */
  native_thread_manager(thread_registry & registry) 
      : registry_{registry}
  { registry_.register_thread(); }

  /**
  \brief Deregisters current thread from the registry.
  */
  ~native_thread_manager() { 
    registry_.deregister_thread(); 
  }

private:
  thread_registry & registry_;
};

/** 
 \brief Native parallel execution policy.
 This policy uses ISO C++ threads as implementation building block allowing
 usage in any ISO C++ compliant platform.
 */
class parallel_execution_native {
public:

  /** 
  \brief Default construct a native parallel execution policy.

  Creates a parallel execution native object.

  The concurrency degree is determined by the platform.

  \note The concurrency degree is fixed to 2 times the hardware concurrency
   degree.
  */
  parallel_execution_native() noexcept :
      parallel_execution_native{
          static_cast<int>(2 * std::thread::hardware_concurrency()), 
          true}
  {}

  /** 
  \brief Constructs a native parallele execution policy.

  Creates a parallel execution native object selecting the concurrency degree
  and ordering mode.
  \param concurrency_degree Number of threads used for parallel algorithms.
  \param order Whether ordered executions is enabled or disabled.
  */
  parallel_execution_native(int concurrency_degree, bool ordering=true) noexcept :
    concurrency_degree_{concurrency_degree},
    ordering_{ordering}
  {
    pool.initialise(concurrency_degree_);
  }

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
  \brief Get a manager object for registration/deregistration in the
  thread index table for current thread.
  */
  native_thread_manager thread_manager() { 
    return native_thread_manager{thread_registry_}; 
  }

  /**
  \brief Get index of current thread in the thread table
  \pre The current thread is currently registered.
  */
  int get_thread_id() const noexcept {
    return thread_registry_.current_index();
  }
  
  /**
  \brief Sets the attributes for the queues built through make_queue<T<(()
  */
  void set_queue_attributes(int size, queue_mode mode) noexcept {
    queue_size_ = size;
    queue_mode_ = mode;
  }

  /**
  \brief Makes a communication queue for elements of type T.
  Constructs a queue using the attributes that can be set via 
  et_queue_attributes(). The value is returned via move semantics.
  */
  template <typename T>
  mpmc_queue<T> make_queue() const {
    return {queue_size_, queue_mode_};
  }

public: 
  /**
  \brief Thread pool for lanching workers.
  \note This member is temporary and is likely to be deprecated or even removed 
        in a future version of GrPPI.
  */
  thread_pool pool;

private: 
  thread_registry thread_registry_;

  int concurrency_degree_;
  bool ordering_;

  constexpr static int default_queue_size = 100;
  int queue_size_ = default_queue_size;

  queue_mode queue_mode_ = queue_mode::blocking;
};

/**
\brief Metafunction that determines if type E is parallel_execution_native
\tparam Execution policy type.
*/
template <typename E>
constexpr bool is_parallel_execution_native() {
  return std::is_same<E, parallel_execution_native>::value;
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
constexpr bool is_supported<parallel_execution_native>() {
  return true;
}


} // end namespace grppi


#endif
