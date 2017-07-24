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

class native_thread_table {
public:
  native_thread_table() noexcept = default;

  void register_thread() noexcept;
  void deregister_thread() noexcept;
  int current_index() const noexcept;
private:
  mutable std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
  std::vector<std::thread::id> ids_;
};

inline void native_thread_table::register_thread() noexcept 
{
  using namespace std;
  while (lock_.test_and_set(memory_order_acquire)) {}
  auto this_id = this_thread::get_id();
  ids_.push_back(this_id);
  lock_.clear(memory_order_release);
}

inline void native_thread_table::deregister_thread() noexcept
{
  using namespace std;
  while (lock_.test_and_set(memory_order_acquire)) {}
  auto this_id = this_thread::get_id();
  ids_.erase(remove(begin(ids_), end(ids_), this_id), end(ids_));
  lock_.clear(memory_order_release);
}

inline int native_thread_table::current_index() const noexcept
{
  using namespace std;
  while (lock_.test_and_set(memory_order_acquire)) {}
  auto this_id = this_thread::get_id();
  auto current = find(begin(ids_), end(ids_), this_id);
  auto index = distance(begin(ids_), current);
  lock_.clear(memory_order_release);
  return index;
};

class native_thread_manager {
public:
  native_thread_manager(native_thread_table & table) 
      : table_{table}
  { table_.register_thread(); }
  ~native_thread_manager() { table_.deregister_thread(); }
private:
  native_thread_table & table_;
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
    return native_thread_manager{thread_table_}; 
  }

  /**
  \brief Get index of current thread in the thread table
  \pre The current thread is currently registered.
  */
  int get_thread_id() {
    return thread_table_.current_index();
  }
  
  void set_queue_attributes(int size, queue_mode mode) {
    queue_size = size;
    lockfree = mode;
  }

  public: 
    thread_pool pool;
    constexpr static int default_queue_size = 100;
    int queue_size = default_queue_size;
    queue_mode lockfree = queue_mode::blocking;

  private: 
    native_thread_table thread_table_;

    int concurrency_degree_;
    bool ordering_;
};

/// Determine if a type is a threading execution policy.
template <typename E>
constexpr bool is_parallel_execution_native() {
  return std::is_same<E, parallel_execution_native>::value;
}

template <typename E>
constexpr bool is_supported();

template <>
constexpr bool is_supported<parallel_execution_native>() {
  return true;
}


} // end namespace grppi


#endif
