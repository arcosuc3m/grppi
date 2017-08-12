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

#ifndef GRPPI_NATIVE_PARALLEL_EXECUTION_NATIVE_H
#define GRPPI_NATIVE_PARALLEL_EXECUTION_NATIVE_H

#include "worker_pool.h"
#include "../common/mpmc_queue.h"
#include "../common/iterator.h"

#include <thread>
#include <atomic>
#include <algorithm>
#include <vector>
#include <type_traits>
#include <tuple>

namespace grppi {

/**
\brief Thread index table to provide portable natural thread indices.

A thread table provides a simple way to offer thread indices (starting from 0).

When a thread registers itself in the registry, its id is added to the vector 
of identifiers. When a thread deregisters itself from the registry its entry
is modified to contain the empty thread id.

To get an integer index, users may call `current_index`, which provides the order
number of the calling thread in the registry.

\note This class is thread safe by means of using a spin-lock.
*/
class thread_registry {
public:
  thread_registry() noexcept = default;

  /**
  \brief Adds the current thread id in the registry.
  */
  void register_thread() noexcept;

  /**
  \brief Removes current thread id from the registry.
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
\brief RAII class to manage registration/deregistration pairs.
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
  \brief Constructs a native parallel execution policy.

  Creates a parallel execution native object selecting the concurrency degree
  and ordering mode.
  \param concurrency_degree Number of threads used for parallel algorithms.
  \param order Whether ordered executions is enabled or disabled.
  */
  parallel_execution_native(int concurrency_degree, bool ordering=true) noexcept :
    concurrency_degree_{concurrency_degree},
    ordering_{ordering}
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
  \brief Get a manager object for registration/deregistration in the
  thread index table for current thread.
  */
  native_thread_manager thread_manager() const { 
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
  \brief Sets the attributes for the queues built through make_queue<T>()
  */
  void set_queue_attributes(int size, queue_mode mode) noexcept {
    queue_size_ = size;
    queue_mode_ = mode;
  }

  /**
  \brief Makes a communication queue for elements of type T.
  Constructs a queue using the attributes that can be set via 
  set_queue_attributes(). The value is returned via move semantics.
  \tparam T Element type for the queue.
  */
  template <typename T>
  mpmc_queue<T> make_queue() const {
    return {queue_size_, queue_mode_};
  }

  /**
  \brief Applies a trasnformation to multiple sequences leaving the result in
  another sequence by chunks according to concurrency degree.
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
      std::size_t sequence_size, Transformer transform_op);

  /**
  \brief Applies a reduction to a sequence of data items. 
  \tparam InputIterator Iterator type for the input sequence.
  \tparam Identity Type for the identity value.
  \tparam Combiner Callable object type for the combination.
  \param first Iterator to the first element of the sequence.
  \param last Iterator to one past the end of the sequence.
  \param identity Identity value for the reduction.
  \param combine_op Combination callable object.
  \pre Iterators in the range `[first,last)` are valid. 
  \return The reduction result
  */
  template <typename InputIterator, typename Identity, typename Combiner>
  auto reduce(InputIterator first, InputIterator last, Identity && identity,
              Combiner && combine_op) const;

  /**
  \brief Applies a map/reduce operation to a sequence of data items.
  \tparam InputIterator Iterator type for the input sequence.
  \tparam Identity Type for the identity value.
  \tparam Transformer Callable object type for the transformation.
  \tparam Combiner Callable object type for the combination.
  \param first Iterator to the first element of the sequence.
  \param last Iterator to one past the end of the sequence.
  \param identity Identity value for the reduction.
  \param transform_op Transformation callable object.
  \param combine_op Combination callable object.
  \pre Iterators in the range `[first,last)` are valid. 
  \return The map/reduce result.
  */
  template <typename ... InputIterators, typename Identity, 
            typename Transformer, typename Combiner>
  auto map_reduce(std::tuple<InputIterators...> firsts, 
                  std::size_t sequence_size,
                  Identity && identity,
                  Transformer && transform_op, Combiner && combine_op) const;

  /**
  \brief Applies a stencil to multiple sequences leaving the result in
  another sequence.
  \tparam InputIterators Iterator types for input sequences.
  \tparam OutputIterator Iterator type for the output sequence.
  \tparam StencilTransformer Callable object type for the stencil transformation.
  \tparam Neighbourhood Callable object for generating neighbourhoods.
  \param firsts Tuple of iterators to input sequences.
  \param first_out Iterator to the output sequence.
  \param sequence_size Size of the input sequences.
  \param transform_op Stencil transformation callable object.
  \param neighbour_op Neighbourhood callable object.
  \pre For every I iterators in the range 
       `[get<I>(firsts), next(get<I>(firsts),sequence_size))` are valid.
  \pre Iterators in the range `[first_out, next(first_out,sequence_size)]` are valid.
  */
  template <typename ... InputIterators, typename OutputIterator,
            typename StencilTransformer, typename Neighbourhood>
  void stencil(std::tuple<InputIterators...> firsts, OutputIterator first_out,
               std::size_t sequence_size,
               StencilTransformer && transform_op,
               Neighbourhood && neighbour_op) const;

private: 
  mutable thread_registry thread_registry_;

  int concurrency_degree_;
  bool ordering_;

  constexpr static int default_queue_size = 100;
  int queue_size_ = default_queue_size;

  queue_mode queue_mode_ = queue_mode::blocking;
};

template <typename ... InputIterators, typename OutputIterator, 
          typename Transformer>
void parallel_execution_native::apply_map(
    std::tuple<InputIterators...> firsts,
    OutputIterator first_out, 
    std::size_t sequence_size, Transformer transform_op)
{
  using namespace std;

  auto process_chunk =
    [transform_op](tuple<InputIterators...> fins, int size, OutputIterator fout)
  {
    const auto l = next(get<0>(fins), size);
    while (get<0>(fins)!=l) {
      *fout++ = apply_deref_increment(transform_op, fins);
    }
  };

  worker_pool workers{concurrency_degree_};
  const int chunk_size = sequence_size / concurrency_degree_;
  
  // One chunk per thread
  for (int i=0; i!=concurrency_degree_-1; ++i) {
    const int delta = chunk_size * i;
    const auto chunk_firsts = iterators_next(firsts,delta);
    const auto chunk_first_out = next(first_out, delta);
    workers.launch(*this, process_chunk, chunk_firsts, chunk_size, chunk_first_out);
  }

  // Last chunk in main thread
  const int delta = chunk_size * (concurrency_degree_ - 1);
  const auto chunk_firsts = iterators_next(firsts,delta);
  const auto chunk_first_out = next(first_out, delta);
  process_chunk(chunk_firsts, sequence_size - delta, chunk_first_out);

  workers.wait();
}

template <typename InputIterator, typename Identity, typename Combiner>
auto parallel_execution_native::reduce(
    InputIterator first, InputIterator last, 
    Identity && identity,
    Combiner && combine_op) const
{
  worker_pool workers{concurrency_degree_};
  auto sequence_size = std::distance(first,last);
  auto chunk_size = sequence_size / concurrency_degree_;

  using result_type = std::decay_t<Identity>;
  std::vector<result_type> partial_results(concurrency_degree_);

  sequential_execution seq;
  auto reduce_chunk = [&](InputIterator f, InputIterator l, std::size_t id) {
    partial_results[id] = seq.reduce(f,l, std::forward<Identity>(identity), 
        std::forward<Combiner>(combine_op));
  };

  for (int i=0; i<concurrency_degree_-1; ++i) {
    auto delta = chunk_size * i;
    auto begin = std::next(first,delta);
    auto end = std::next(begin, chunk_size);

    workers.launch(*this, reduce_chunk, begin, end, i);
  }

  auto delta = chunk_size * (concurrency_degree_-1);
  auto begin = std::next(first, delta);
  reduce_chunk(begin, last, concurrency_degree_-1);

  workers.wait();

  return seq.reduce(std::next(partial_results.begin()), partial_results.end(), 
      partial_results[0], combine_op);
}

template <typename ... InputIterators, typename Identity, 
          typename Transformer, typename Combiner>
auto parallel_execution_native::map_reduce(
    std::tuple<InputIterators...> firsts, 
    std::size_t sequence_size,
    Identity && identity,
    Transformer && transform_op, Combiner && combine_op) const
{
  using result_type = std::decay_t<Identity>;
  std::vector<result_type> partial_results(concurrency_degree_);

  sequential_execution seq;
  auto map_reduce_chunk = [&](auto f, std::size_t sz, std::size_t id) {
    partial_results[id] = seq.map_reduce(f, sz,
        std::forward<Identity>(partial_results[id]), 
        std::forward<Transformer>(transform_op), std::forward<Combiner>(combine_op));
  };

  const auto chunk_size = sequence_size/concurrency_degree_;

  worker_pool workers{concurrency_degree_};
  for(int i=0;i<concurrency_degree_-1;++i){    
    auto delta = chunk_size * i;
    auto begin = iterators_next(firsts,delta);

    workers.launch(*this, map_reduce_chunk, begin, chunk_size, i);
  }

  auto delta = chunk_size * (concurrency_degree_-1);
  auto begin = iterators_next(firsts, delta);
  partial_results[concurrency_degree_-1] = seq.map_reduce(begin, 
        sequence_size - delta,
        partial_results[concurrency_degree_-1], 
        std::forward<Transformer>(transform_op), 
        std::forward<Combiner>(combine_op));

  workers.wait();

  return seq.reduce(std::next(partial_results.begin()), partial_results.end(),
      partial_results[0],
      std::forward<Combiner>(combine_op));
}

template <typename ... InputIterators, typename OutputIterator,
          typename StencilTransformer, typename Neighbourhood>
void parallel_execution_native::stencil(
    std::tuple<InputIterators...> firsts, OutputIterator first_out,
    std::size_t sequence_size,
    StencilTransformer && transform_op,
    Neighbourhood && neighbour_op) const
{
  sequential_execution seq;
  auto process_chunk =
    [&transform_op, &neighbour_op,seq](auto fins, std::size_t sz, auto fout)
  {
    seq.stencil(fins, fout, sz,
      std::forward<StencilTransformer>(transform_op),
      std::forward<Neighbourhood>(neighbour_op));
  };

  worker_pool workers{concurrency_degree_};

  const int chunk_size = sequence_size / concurrency_degree_;
  for (int i=0; i!=concurrency_degree_-1; ++i) {
    const int delta = chunk_size * i;
    const auto chunk_firsts = iterators_next(firsts,delta);
    const auto chunk_out = std::next(first_out,delta);
    workers.launch(*this, process_chunk, chunk_firsts, chunk_size, chunk_out);
  }

  const int delta = chunk_size * (concurrency_degree_ - 1);
  const auto chunk_firsts = iterators_next(firsts,delta);
  const auto chunk_out = std::next(first_out,delta);
  process_chunk(chunk_firsts, sequence_size - delta, chunk_out);

  workers.wait();
}

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
