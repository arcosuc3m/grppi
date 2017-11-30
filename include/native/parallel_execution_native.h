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
#include "../common/execution_traits.h"
#include "../common/splitter_queue.h"
#include "../common/split_consumer_queue.h"
#include "../common/joiner_queue.h"
#include "../common/windower_queue.h"


#include <thread>
#include <atomic>
#include <algorithm>
#include <vector>
#include <type_traits>
#include <tuple>
#include <experimental/optional>

//#include "extrae_user_events.h"
//#include <extrae.h>
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

  parallel_execution_native(const parallel_execution_native & ex) :
      parallel_execution_native{ex.concurrency_degree_, ex.ordering_}
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
  \brief Makes a special communication queue that splits the data stream
  into a given number of consumer queues.
  Constructs the internal queues using the attributes that can be set via 
  set_queue_attributes(). The value is returned via move semantics.
  \tparam Queue Queue type for the consumer queues.
  \tparam Policy Policy type for the splitting management.
  \param q Input queue to be splited in several streams
  \param num_queue number of consumers queues
  \param policy splitting policy  
  */
  template <typename Queue, typename Policy>
  splitter_queue<typename Queue::value_type, Queue, Policy> make_split_queue (
      Queue & q, int num_queues, Policy policy) const
  {
     return {q, num_queues, policy, queue_size_, queue_mode_};
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
  void map(std::tuple<InputIterators...> firsts,
      OutputIterator first_out, 
      std::size_t sequence_size, Transformer transform_op) const;

  /**
  \brief Applies a reduction to a sequence of data items. 
  \tparam InputIterator Iterator type for the input sequence.
  \tparam Identity Type for the identity value.
  \tparam Combiner Callable object type for the combination.
  \param first Iterator to the first element of the sequence.
  \param sequence_size Size of the input sequence.
  \param identity Identity value for the reduction.
  \param combine_op Combination callable object.
  \pre Iterators in the range `[first,last)` are valid. 
  \return The reduction result
  */
  template <typename InputIterator, typename Identity, typename Combiner>
  auto reduce(InputIterator first, std::size_t sequence_size, 
              Identity && identity, Combiner && combine_op) const;

  /**
  \brief Applies a map/reduce operation to a sequence of data items.
  \tparam InputIterator Iterator type for the input sequence.
  \tparam Identity Type for the identity value.
  \tparam Transformer Callable object type for the transformation.
  \tparam Combiner Callable object type for the combination.
  \param first Iterator to the first element of the sequence.
  \param sequence_size Size of the input sequence.
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

  /**
  \brief Invoke \ref md_divide-conquer.
  \tparam Input Type used for the input problem.
  \tparam Divider Callable type for the divider operation.
  \tparam Solver Callable type for the solver operation.
  \tparam Combiner Callable type for the combiner operation.
  \param ex Sequential execution policy object.
  \param input Input problem to be solved.
  \param divider_op Divider operation.
  \param solver_op Solver operation.
  \param combine_op Combiner operation.
  */
  template <typename Input, typename Divider, typename Solver, typename Combiner>
  auto divide_conquer(Input && input, 
                      Divider && divide_op, 
                      Solver && solve_op, 
                      Combiner && combine_op) const; 

  /**
  \brief Invoke \ref md_pipeline.
  \tparam Generator Callable type for the generator operation.
  \tparam Transformers Callable types for the transformers in the pipeline.
  \param generate_op Generator operation.
  \param transform_ops Transformer operations.
  */
  template <typename Generator, typename ... Transformers>
  void pipeline(Generator && generate_op, 
                Transformers && ... transform_ops) const;

  /**
  \brief Invoke \ref md_stream_pool.
  \tparam Population Type for the initial population.
  \tparam Selection Callable type for the selection operation.
  \tparam Selection Callable type for the evolution operation.
  \tparam Selection Callable type for the evaluation operation.
  \tparam Selection Callable type for the termination operation.
  \param population initial population.
  \param selection_op Selection operation.
  \param evolution_op Evolution operations.
  \param eval_op Evaluation operation.
  \param termination_op Termination operation.
  */
  template <typename Population, typename Selection, typename Evolution,
            typename Evaluation, typename Predicate>
  void stream_pool(Population & population,
                Selection && selection_op,
                Evolution && evolve_op,
                Evaluation && eval_op,
                Predicate && termination_op) const;
private:

  template <typename Input, typename Divider, typename Solver, typename Combiner>
  auto divide_conquer(Input && input, 
                      Divider && divide_op, 
                      Solver && solve_op, 
                      Combiner && combine_op,
                      std::atomic<int> & num_threads) const; 

  template <typename Queue, typename Consumer,
            requires_no_pattern<Consumer> = 0,
            requires_no_join_queue<Consumer> = 0>
  void do_pipeline(Queue & input_queue, Consumer && consume_op) const;

  template <typename Queue, typename JoinQueue,
          requires_no_pattern<JoinQueue> = 0,
          requires_join_queue<JoinQueue> = 0>
  void do_pipeline(
    Queue & input_queue,
    std::tuple<JoinQueue &, int> & join_queue) const;

  template <typename Queue, typename Transformer, typename ... OtherTransformers,
            requires_no_pattern<Transformer> = 0>
  void do_pipeline(Queue & input_queue, Transformer && transform_op,
      OtherTransformers && ... other_ops) const;

  template <typename Queue, typename FarmTransformer,
            template <typename> class Farm,
            requires_farm<Farm<FarmTransformer>> = 0>
  void do_pipeline(Queue & input_queue, 
      Farm<FarmTransformer> & farm_obj) const
  {
    do_pipeline(input_queue, std::move(farm_obj));
  }

  template <typename Queue, typename FarmTransformer,
            template <typename> class Farm,
            requires_farm<Farm<FarmTransformer>> = 0>
  void do_pipeline( Queue & input_queue, 
      Farm<FarmTransformer> && farm_obj) const;

  template <typename Queue, typename Transformer, typename Window,
          template <typename C, typename W> class Farm,
          typename ... OtherTransformers,
          requires_window_farm< Farm< Transformer, Window >> = 0>
  void do_pipeline(
    Queue && input_queue,
    Farm<Transformer,Window> & farm_obj,
    OtherTransformers && ... other_transform_ops) const
  {
    do_pipeline(input_queue, std::move(farm_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);

  }

  template <typename Queue, typename Transformer, typename Window,
          template <typename C, typename W> class Farm,
          typename ... OtherTransformers,
          requires_window_farm<Farm<Transformer,Window>> = 0>
  void do_pipeline(
    Queue && input_queue,
    Farm<Transformer,Window> && farm_obj,
    OtherTransformers && ... other_transform_ops) const;


  template <typename Queue, typename Policy, typename ... Transformers,
          template <typename P> class Window,
          typename ... OtherTransformers,
          requires_window< Window <Policy>> = 0 >
  void do_pipeline(
    Queue & input_queue,
    Window<Policy> && win_obj,
    OtherTransformers & ... other_transform_ops) const
  {
    do_pipeline(input_queue, std::move(win_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename Queue, typename Policy, typename ... Transformers,
          template <typename P> class Window,
          typename ... OtherTransformers,
          requires_window< Window <Policy>> = 0 >
  void do_pipeline(
    Queue && input_queue,
    Window<Policy> && win_obj,
    OtherTransformers && ... other_transform_ops) const;

  template <typename Queue, typename Policy, typename ... Transformers,
          template <typename P> class Window,
          typename ... OtherTransformers,
          requires_active_window< Window <Policy>> = 0 >
  void do_pipeline(
    Queue && input_queue,
    Window<Policy> & win_obj,
    OtherTransformers && ... other_transform_ops) const
  {
    do_pipeline(input_queue, std::move(win_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename Queue, typename Policy, typename ... Transformers,
          template <typename P> class Window,
          typename ... OtherTransformers,
          requires_active_window< Window <Policy>> = 0 >
  void do_pipeline(
    Queue && input_queue,
    Window<Policy> && win_obj,
    OtherTransformers && ... other_transform_ops) const;

 
  template <std::size_t index, typename InQueue, typename OutQueue,
          typename ...Transformers, typename Policy,
          template <typename P, typename ... T> class SplitJoin,
          requires_split_join< SplitJoin <Policy, Transformers...>> = 0>
  typename std::enable_if<(index == (sizeof...(Transformers)-1)),void>::type
    create_flow(
          InQueue && split_queue,  OutQueue && join_queue,
          SplitJoin<Policy,Transformers...> && split_obj, std::vector<std::thread> & tasks) const;

  template <std::size_t index, typename InQueue, typename OutQueue,
          typename ...Transformers, typename Policy,
          template <typename P, typename ... T> class SplitJoin,
          requires_split_join< SplitJoin <Policy, Transformers...>> = 0>
  typename std::enable_if<(index != (sizeof...(Transformers)-1)),void>::type
    create_flow(
          InQueue && split_queue,  OutQueue && join_queue,
          SplitJoin<Policy,Transformers...> && split_obj,std::vector<std::thread> & tasks) const;


  template <typename Queue, typename Policy, typename ... Transformers,
          template <typename P, typename ... T> class SplitJoin,
          typename ... OtherTransformers,
          requires_split_join< SplitJoin <Policy, Transformers...>> = 0 >
  void do_pipeline(
    Queue && input_queue,
    SplitJoin<Policy,Transformers...> & split_obj,
    OtherTransformers && ... other_transform_ops) const
  {
    do_pipeline(input_queue, std::move(split_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);

  }

  template <typename Queue, typename Policy, typename ... Transformers,
          template <typename P, typename ... T> class SplitJoin,
          typename ... OtherTransformers,
          requires_split_join< SplitJoin <Policy, Transformers...>> = 0 >
  void do_pipeline(
    Queue && input_queue,
    SplitJoin<Policy,Transformers...> && split_obj,
    OtherTransformers && ... other_transform_ops) const;



  template <typename Queue, typename FarmTransformer, 
            template <typename> class Farm,
            typename ... OtherTransformers,
            requires_farm<Farm<FarmTransformer>> = 0>
  void do_pipeline(Queue & input_queue, 
      Farm<FarmTransformer> & farm_obj,
      OtherTransformers && ... other_transform_ops) const
  {
    do_pipeline(input_queue, std::move(farm_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename Queue, typename FarmTransformer, 
            template <typename> class Farm,
            typename ... OtherTransformers,
            requires_farm<Farm<FarmTransformer>> = 0>
  void do_pipeline(Queue & input_queue, 
      Farm<FarmTransformer> && farm_obj,
      OtherTransformers && ... other_transform_ops) const;

  template <typename Queue, typename Predicate, 
            template <typename> class Filter,
            typename ... OtherTransformers,
            requires_filter<Filter<Predicate>> =0>
  void do_pipeline(Queue & input_queue, 
      Filter<Predicate> & filter_obj,
      OtherTransformers && ... other_transform_ops) const
  {
    do_pipeline(input_queue, std::move(filter_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename Queue, typename Predicate, 
            template <typename> class Filter,
            typename ... OtherTransformers,
            requires_filter<Filter<Predicate>> =0>
  void do_pipeline(Queue & input_queue, 
      Filter<Predicate> && farm_obj,
      OtherTransformers && ... other_transform_ops) const;

  template <typename Queue, typename Combiner, typename Identity,
            template <typename C, typename I> class Reduce,
            typename ... OtherTransformers,
            requires_reduce<Reduce<Combiner,Identity>> = 0>
  void do_pipeline(Queue && input_queue, Reduce<Combiner,Identity> & reduce_obj,
                   OtherTransformers && ... other_transform_ops) const
  {
    do_pipeline(input_queue, std::move(reduce_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  };

  template <typename Queue, typename Combiner, typename Identity,
            template <typename C, typename I> class Reduce,
            typename ... OtherTransformers,
            requires_reduce<Reduce<Combiner,Identity>> = 0>
  void do_pipeline(Queue && input_queue, Reduce<Combiner,Identity> && reduce_obj,
                   OtherTransformers && ... other_transform_ops) const;

  template <typename Queue, typename Transformer, typename Predicate,
            template <typename T, typename P> class Iteration,
            typename ... OtherTransformers,
            requires_iteration<Iteration<Transformer,Predicate>> =0,
            requires_no_pattern<Transformer> =0>
  void do_pipeline(Queue & input_queue, Iteration<Transformer,Predicate> & iteration_obj,
                   OtherTransformers && ... other_transform_ops) const
  {
    do_pipeline(input_queue, std::move(iteration_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename Queue, typename Transformer, typename Predicate,
            template <typename T, typename P> class Iteration,
            typename ... OtherTransformers,
            requires_iteration<Iteration<Transformer,Predicate>> =0,
            requires_no_pattern<Transformer> =0>
  void do_pipeline(Queue & input_queue, Iteration<Transformer,Predicate> && iteration_obj,
                   OtherTransformers && ... other_transform_ops) const;

  template <typename Queue, typename Transformer, typename Predicate,
            template <typename T, typename P> class Iteration,
            typename ... OtherTransformers,
            requires_iteration<Iteration<Transformer,Predicate>> =0,
            requires_pipeline<Transformer> =0>
  void do_pipeline(Queue & input_queue, Iteration<Transformer,Predicate> && iteration_obj,
                   OtherTransformers && ... other_transform_ops) const;


  template <typename Queue, typename ... Transformers,
            template <typename...> class Pipeline,
            requires_pipeline<Pipeline<Transformers...>> = 0>
  void do_pipeline(Queue & input_queue,
      Pipeline<Transformers...> & pipeline_obj) const
  {
    do_pipeline(input_queue, std::move(pipeline_obj));
  }

  template <typename Queue, typename ... Transformers,
            template <typename...> class Pipeline,
            requires_pipeline<Pipeline<Transformers...>> = 0>
  void do_pipeline(Queue & input_queue,
      Pipeline<Transformers...> && pipeline_obj) const;

  template <typename Queue, typename ... Transformers,
            template <typename...> class Pipeline,
            typename ... OtherTransformers,
            requires_pipeline<Pipeline<Transformers...>> = 0>
  void do_pipeline(Queue & input_queue,
      Pipeline<Transformers...> & pipeline_obj,
      OtherTransformers && ... other_transform_ops) const
  {
    do_pipeline(input_queue, std::move(pipeline_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename Queue, typename ... Transformers,
            template <typename...> class Pipeline,
            typename ... OtherTransformers,
            requires_pipeline<Pipeline<Transformers...>> = 0>
  void do_pipeline(Queue & input_queue,
      Pipeline<Transformers...> && pipeline_obj,
      OtherTransformers && ... other_transform_ops) const;

  template <typename Queue, typename ... Transformers,
            std::size_t ... I>
  void do_pipeline_nested(
      Queue & input_queue, 
      std::tuple<Transformers...> && transform_ops,
      std::index_sequence<I...>) const;

private: 
  mutable thread_registry thread_registry_;

  int concurrency_degree_;
  bool ordering_;

  constexpr static int default_queue_size = 100;
  int queue_size_ = default_queue_size;

  queue_mode queue_mode_ = queue_mode::blocking;
//  queue_mode queue_mode_ = queue_mode::lockfree;
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
\brief Determines if an execution policy is supported in the current compilation.
\note Specialization for parallel_execution_native.
*/
template <>
constexpr bool is_supported<parallel_execution_native>() { return true; }

/**
\brief Determines if an execution policy supports the map pattern.
\note Specialization for parallel_execution_native.
*/
template <>
constexpr bool supports_map<parallel_execution_native>() { return true; }

/**
\brief Determines if an execution policy supports the reduce pattern.
\note Specialization for parallel_execution_native.
*/
template <>
constexpr bool supports_reduce<parallel_execution_native>() { return true; }

/**
\brief Determines if an execution policy supports the map-reduce pattern.
\note Specialization for parallel_execution_native.
*/
template <>
constexpr bool supports_map_reduce<parallel_execution_native>() { return true; }

/**
\brief Determines if an execution policy supports the stencil pattern.
\note Specialization for parallel_execution_native.
*/
template <>
constexpr bool supports_stencil<parallel_execution_native>() { return true; }

/**
\brief Determines if an execution policy supports the divide/conquer pattern.
\note Specialization for parallel_execution_native.
*/
template <>
constexpr bool supports_divide_conquer<parallel_execution_native>() { return true; }

/**
\brief Determines if an execution policy supports the pipeline pattern.
\note Specialization for parallel_execution_native.
*/
template <>
constexpr bool supports_pipeline<parallel_execution_native>() { return true; }

/**
\brief Determines if an execution policy supports the stream pool pattern.
\note Specialization for parallel_execution_native.
*/
template <>
constexpr bool supports_stream_pool<parallel_execution_native>() { return true; }

/**
\brief Determines if an execution policy supports the split join pattern.
\note Specialization for parallel_execution_native.
*/
template <>
constexpr bool supports_split_join<parallel_execution_native>() { return true; }

template <typename ... InputIterators, typename OutputIterator, 
          typename Transformer>
void parallel_execution_native::map(
    std::tuple<InputIterators...> firsts,
    OutputIterator first_out, 
    std::size_t sequence_size, Transformer transform_op) const
{
  using namespace std;

  auto process_chunk =
    [&transform_op](auto fins, std::size_t size, auto fout)
  {
    const auto l = next(get<0>(fins), size);
    while (get<0>(fins)!=l) {
      *fout++ = apply_deref_increment(
          std::forward<Transformer>(transform_op), fins);
    }
  };

  const int chunk_size = sequence_size / concurrency_degree_;
  
  {
    worker_pool workers{concurrency_degree_};
    for (int i=0; i!=concurrency_degree_-1; ++i) {
      const auto delta = chunk_size * i;
      const auto chunk_firsts = iterators_next(firsts,delta);
      const auto chunk_first_out = next(first_out, delta);
      workers.launch(*this, process_chunk, chunk_firsts, chunk_size, chunk_first_out);
    }

    const auto delta = chunk_size * (concurrency_degree_ - 1);
    const auto chunk_firsts = iterators_next(firsts,delta);
    const auto chunk_first_out = next(first_out, delta);
    process_chunk(chunk_firsts, sequence_size - delta, chunk_first_out);
  } // Pool synch
}

template <typename InputIterator, typename Identity, typename Combiner>
auto parallel_execution_native::reduce(
    InputIterator first, std::size_t sequence_size,
    Identity && identity,
    Combiner && combine_op) const
{
  using result_type = std::decay_t<Identity>;
  std::vector<result_type> partial_results(concurrency_degree_);

  constexpr sequential_execution seq;
  auto process_chunk = [&](InputIterator f, std::size_t sz, std::size_t id) {
    partial_results[id] = seq.reduce(f,sz, std::forward<Identity>(identity), 
        std::forward<Combiner>(combine_op));
  };

  const auto chunk_size = sequence_size / concurrency_degree_;

  { 
    worker_pool workers{concurrency_degree_};
    for (int i=0; i<concurrency_degree_-1; ++i) {
      const auto delta = chunk_size * i;
      const auto chunk_first = std::next(first,delta);
      workers.launch(*this, process_chunk, chunk_first, chunk_size, i);
    }

    const auto delta = chunk_size * (concurrency_degree_-1);
    const auto chunk_first = std::next(first, delta);
    const auto chunk_sz = sequence_size - delta;
    process_chunk(chunk_first, chunk_sz, concurrency_degree_-1);
  } // Pool synch

  return seq.reduce(std::next(partial_results.begin()), 
      partial_results.size()-1, std::forward<result_type>(partial_results[0]), 
      std::forward<Combiner>(combine_op));
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

  constexpr sequential_execution seq;
  auto process_chunk = [&](auto f, std::size_t sz, std::size_t id) {
    partial_results[id] = seq.map_reduce(f, sz,
        std::forward<Identity>(partial_results[id]), 
        std::forward<Transformer>(transform_op), 
        std::forward<Combiner>(combine_op));
  };

  const auto chunk_size = sequence_size / concurrency_degree_;

  {
    worker_pool workers{concurrency_degree_};
    for(int i=0;i<concurrency_degree_-1;++i){    
      const auto delta = chunk_size * i;
      const auto chunk_firsts = iterators_next(firsts,delta);
      workers.launch(*this, process_chunk, chunk_firsts, chunk_size, i);
    }

    const auto delta = chunk_size * (concurrency_degree_-1);
    const auto chunk_firsts = iterators_next(firsts, delta);
    process_chunk(chunk_firsts, sequence_size - delta, concurrency_degree_-1);
  } // Pool synch

  return seq.reduce(std::next(partial_results.begin()), 
     partial_results.size()-1, std::forward<result_type>(partial_results[0]), 
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
  constexpr sequential_execution seq;
  auto process_chunk =
    [&transform_op, &neighbour_op,seq](auto fins, std::size_t sz, auto fout)
  {
    seq.stencil(fins, fout, sz,
      std::forward<StencilTransformer>(transform_op),
      std::forward<Neighbourhood>(neighbour_op));
  };

  const auto chunk_size = sequence_size / concurrency_degree_;
  {
    worker_pool workers{concurrency_degree_};

    for (int i=0; i!=concurrency_degree_-1; ++i) {
      const auto delta = chunk_size * i;
      const auto chunk_firsts = iterators_next(firsts,delta);
      const auto chunk_out = std::next(first_out,delta);
      workers.launch(*this, process_chunk, chunk_firsts, chunk_size, chunk_out);
    }

    const auto delta = chunk_size * (concurrency_degree_ - 1);
    const auto chunk_firsts = iterators_next(firsts,delta);
    const auto chunk_out = std::next(first_out,delta);
    process_chunk(chunk_firsts, sequence_size - delta, chunk_out);
  } // Pool synch
}

template <typename Input, typename Divider, typename Solver, typename Combiner>
auto parallel_execution_native::divide_conquer(
    Input && problem, 
    Divider && divide_op, 
    Solver && solve_op, 
    Combiner && combine_op) const
{
  std::atomic<int> num_threads{concurrency_degree_-1};

  return divide_conquer(std::forward<Input>(problem), std::forward<Divider>(divide_op), 
        std::forward<Solver>(solve_op), std::forward<Combiner>(combine_op),
        num_threads);
}

template <typename Generator, typename ... Transformers>
void parallel_execution_native::pipeline(
    Generator && generate_op, 
    Transformers && ... transform_ops) const
{
  using namespace std;
  using result_type = decay_t<typename result_of<Generator()>::type>;
  using output_type = pair<result_type,long>;
  auto output_queue = make_queue<output_type>();

  thread generator_task([&,this]() {
    auto manager = thread_manager();

    long order = 0;
    for (;;) {
      auto item{generate_op()};
      output_queue.push(make_pair(item, order));
      order++;
      if (!item) break;
    }
  });

  do_pipeline(output_queue, forward<Transformers>(transform_ops)...);
  generator_task.join();
}

// PRIVATE MEMBERS
template <typename Population, typename Selection, typename Evolution,
            typename Evaluation, typename Predicate>
void parallel_execution_native::stream_pool(Population & population,
                Selection && selection_op,
                Evolution && evolve_op,
                Evaluation && eval_op,
                Predicate && termination_op) const
{

  using namespace std;
  using namespace experimental;

  using selected_type = typename std::result_of<Selection(Population&)>::type;
  using individual_type = typename Population::value_type;
  using selected_op_type = optional<selected_type>;
  using individual_op_type = optional<individual_type>;

  auto selected_queue = make_queue<selected_op_type>();
  auto output_queue = make_queue<individual_op_type>();

      
  std::atomic<bool> end{false};
  std::atomic<int> done_threads{0};
  std::atomic_flag lock = ATOMIC_FLAG_INIT;

  vector<thread> tasks;
  for(auto i = 0; i< concurrency_degree_; i++) 
    tasks.push_back(std::thread( [&](){
    auto selection = selected_queue.pop();
    while(selection){
      auto evolved = evolve_op(*selection);
      auto filtered = eval_op(*selection, evolved);
      if(termination_op(filtered)){ 
        end = true; 
      }
      output_queue.push({filtered});
      selection = selected_queue.pop();
    }
    done_threads++;
    if(done_threads == concurrency_degree_) 
     output_queue.push(individual_op_type{});
  }));
     
  std::thread selector([&](){
    for(;;) {
      if(end) break;
      while(lock.test_and_set());
      if( population.size() != 0 ){
        auto selection = selection_op(population);
        lock.clear();
        selected_queue.push({selection});
      }else{
        lock.clear();
      }
    }
    for(int i=0;i<concurrency_degree_;i++) selected_queue.push(selected_op_type{});
  });
  
 std::thread sink([&](){
    auto item = output_queue.pop();
    while(item) {
      while(lock.test_and_set());
      population.push_back(*item);
      lock.clear();
      item = output_queue.pop();
    }
  });
/*  worker_pool workers{concurrency_degree_};
  tasks.push_back({task});
          
  workers.launch_tasks(*this, task, concurrency_degree_);*/
  sink.join();
  selector.join();
  for(auto i = 0; i< concurrency_degree_; i++) 
    tasks[i].join();
 // workers.wait();

}


template <typename Input, typename Divider, typename Solver, typename Combiner>
auto parallel_execution_native::divide_conquer(
    Input && input, 
    Divider && divide_op, 
    Solver && solve_op, 
    Combiner && combine_op,
    std::atomic<int> & num_threads) const
{
  constexpr sequential_execution seq;
  if (num_threads.load() <=0) {
    return seq.divide_conquer(std::forward<Input>(input), 
        std::forward<Divider>(divide_op), std::forward<Solver>(solve_op), 
        std::forward<Combiner>(combine_op));
  }

  auto subproblems = divide_op(std::forward<Input>(input));
  if (subproblems.size()<=1) { return solve_op(std::forward<Input>(input)); }

  using subresult_type = 
      std::decay_t<typename std::result_of<Solver(Input)>::type>;
  std::vector<subresult_type> partials(subproblems.size()-1);

  auto process_subproblem = [&,this](auto it, std::size_t div) {
    partials[div] = this->divide_conquer(std::forward<Input>(*it), 
        std::forward<Divider>(divide_op), std::forward<Solver>(solve_op), 
        std::forward<Combiner>(combine_op), num_threads);
  };

  int division = 0;

  worker_pool workers{num_threads.load()};
  auto i = subproblems.begin() + 1;
  while (i!=subproblems.end() && num_threads.load()>0) {
    workers.launch(*this,process_subproblem, i++, division++);
    num_threads--;
  }

  while (i!=subproblems.end()) {
    partials[division] = seq.divide_conquer(std::forward<Input>(*i++), 
        std::forward<Divider>(divide_op), std::forward<Solver>(solve_op), 
        std::forward<Combiner>(combine_op));
  }

  auto subresult = divide_conquer(std::forward<Input>(*subproblems.begin()), 
      std::forward<Divider>(divide_op), std::forward<Solver>(solve_op), 
      std::forward<Combiner>(combine_op), num_threads);

  workers.wait();

  return seq.reduce(partials.begin(), partials.size(), 
      std::forward<subresult_type>(subresult), std::forward<Combiner>(combine_op));
}

template <typename Queue, typename JoinQueue,
          requires_no_pattern<JoinQueue> = 0,
          requires_join_queue<JoinQueue> = 0>
void parallel_execution_native::do_pipeline(
    Queue & input_queue,
    std::tuple<JoinQueue&,int> & joiner_tuple) const
{
    std::get<0>(joiner_tuple).add_queue(input_queue,std::get<1>(joiner_tuple));    
    std::get<0>(joiner_tuple).wait();
}


template <typename Queue, typename Consumer,
          requires_no_pattern<Consumer> = 0,
          requires_no_join_queue<Consumer> = 0>
void parallel_execution_native::do_pipeline(
    Queue & input_queue, 
    Consumer && consume_op) const
{
  using namespace std;
  using input_type = typename Queue::value_type;
  using input_value_type = typename input_type::first_type;

  auto manager = thread_manager(); 
  //Extrae_event(60000000,1);
  if (!is_ordered()) {
    for (;;) {
      auto item = input_queue.pop();
      if (!item.first) break;
      consume_op(*item.first);
    }
    return;
  }
  vector<input_type> elements;
  long current = 0;
  for (;;) {
    auto item = input_queue.pop();
    if (!item.first) break;
    if(current == item.second){
      consume_op(*item.first);
      current ++;
    }
    else {
      elements.push_back(item);
    }
    // TODO: Probably find_if() + erase 
    for (auto it=elements.begin(); it!=elements.end(); it++) {
      if(it->second == current) {
        consume_op(*it->first);
        elements.erase(it);
        current++;
        break;
      }
    }
  }
  while (elements.size()>0) {
    // TODO: Probably find_if() + erase
    for (auto it = elements.begin(); it != elements.end(); it++) {
      if(it->second == current) {
        consume_op(*it->first);
        elements.erase(it);
        current++;
        break;
      }
    }
  }
  //Extrae_event(60000000,0);
}

template <typename Queue, typename Transformer, 
          typename ... OtherTransformers,
          requires_no_pattern<Transformer> =0>
void parallel_execution_native::do_pipeline(
    Queue & input_queue, 
    Transformer && transform_op,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
  using namespace experimental;

  using input_item_type = typename Queue::value_type;
  using input_item_value_type = typename input_item_type::first_type::value_type;
  using transform_result_type = 
      decay_t<typename result_of<Transformer(input_item_value_type)>::type>;
  using output_item_value_type = optional<transform_result_type>;
  using output_item_type = pair<output_item_value_type,long>;
  auto output_queue = make_queue<output_item_type>();

  thread task([&,this]() {
    auto manager = thread_manager();

    long order = 0;
    for (;;) {
      auto item{input_queue.pop()};

     // Extrae_event(60000000,2);
      if (!item.first) break;
      auto out = output_item_value_type{transform_op(*item.first)};
      //Extrae_event(60000000,0);
      output_queue.push(make_pair(out, item.second));
    }
    output_queue.push(make_pair(output_item_value_type{},-1));
  });
  do_pipeline(output_queue, 
      forward<OtherTransformers>(other_transform_ops)...);
  task.join();
}

template <typename Queue, typename FarmTransformer,
          template <typename> class Farm,
          requires_farm<Farm<FarmTransformer>> =0>
void parallel_execution_native::do_pipeline(
    Queue & input_queue, 
    Farm<FarmTransformer> && farm_obj) const
{
  using namespace std;
  using input_item_type = typename Queue::value_type;
  using input_item_value_type = typename input_item_type::first_type::value_type;
  using transform_result_type = 
      decay_t<typename result_of<FarmTransformer(input_item_value_type)>::type>;
  using output_item_value_type = experimental::optional<transform_result_type>;
  using output_item_type = pair<output_item_value_type,long>;

  auto farm_task = [&](int nt) {
    long order = 0;
    auto item{input_queue.pop()}; 
    while (item.first) {
      farm_obj(*item.first);
      item = input_queue.pop();
    }
    input_queue.push(item);
  };

  auto ntasks = farm_obj.cardinality();
  worker_pool workers{ntasks};
  workers.launch_tasks(*this, farm_task, ntasks);  
  workers.wait();
}

template <typename Queue, typename FarmTransformer, 
          template <typename> class Farm,
          typename ... OtherTransformers,
          requires_farm<Farm<FarmTransformer>> =0>
void parallel_execution_native::do_pipeline(
    Queue & input_queue, 
    Farm<FarmTransformer> && farm_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
  using namespace experimental;

  using input_item_type = typename Queue::value_type;
  using input_item_value_type = typename input_item_type::first_type::value_type;
  using transform_result_type = 
      decay_t<typename result_of<FarmTransformer(input_item_value_type)>::type>;
  using output_item_value_type = experimental::optional<transform_result_type>;
  using output_item_type = pair<output_item_value_type,long>;

  auto output_queue = make_queue<output_item_type>();
  atomic<int> done_threads{0};
  auto farm_task = [&](int nt) {
   // long order = 0;
    auto item{input_queue.pop()}; 
    while (item.first) {
      auto out = output_item_value_type{farm_obj(*item.first)};
      output_queue.push(make_pair(out,item.second)) ;
      item = input_queue.pop(); 
    }
    input_queue.push(item);
    done_threads++;
    if (done_threads == nt) {
      output_queue.push(make_pair(output_item_value_type{}, -1));
    }
  };

  auto ntasks = farm_obj.cardinality();
  worker_pool workers{ntasks};
  workers.launch_tasks(*this, farm_task, ntasks);  
  do_pipeline(output_queue, 
      forward<OtherTransformers>(other_transform_ops)... );

  workers.wait();
}



template <typename Queue, typename Transformer, typename Window,
          template <typename C, typename W> class Farm,
          typename ... OtherTransformers,
          requires_window_farm<Farm<Transformer,Window>> = 0>
void parallel_execution_native::do_pipeline(
    Queue && input_queue,
    Farm<Transformer,Window> && farm_obj,
    OtherTransformers && ... other_transform_ops) const 
{
  using namespace std;
  using namespace experimental;

  using windower_queue_t = windower_queue<Queue,std::decay_t<Window>>;
  windower_queue<Queue,std::decay_t<Window>> window_queue{input_queue, farm_obj.get_window()};
 
  using optional_window_t = typename windower_queue_t::value_type;
  using window_type = typename optional_window_t::first_type::value_type;
  using result_type = typename result_of<Transformer(window_type)>::type;
  using output_optional_type = experimental::optional <result_type>;
  using output_item_type = pair <output_optional_type,long>;
  
  auto output_queue = make_queue<output_item_type>();
  atomic<int> done_threads{0}; 

  auto farm_task = [&](int nt) {
    long order = 0;
    auto item{window_queue.pop()};
    while (item.first) {
      auto out = output_optional_type{farm_obj.transform(*item.first)};
      output_queue.push(make_pair(out, item.second)) ;
      item = window_queue.pop();
    }
    input_queue.push(make_pair(output_optional_type{}, -1));
    done_threads++;
    if (done_threads == nt) {
      output_queue.push(make_pair(output_optional_type{}, -1));
    }
  };
  
  auto ntasks = farm_obj.cardinality();
  worker_pool workers{ntasks};
  workers.launch_tasks(*this, farm_task, ntasks);

  do_pipeline(output_queue, 
      forward<OtherTransformers>(other_transform_ops)... );
  workers.wait();

}

template <std::size_t index, typename InQueue, typename OutQueue, 
          typename ...Transformers, typename Policy,
          template <typename P, typename ... T> class SplitJoin,
          requires_split_join< SplitJoin <Policy, Transformers...>> = 0>
typename std::enable_if<(index == (sizeof...(Transformers)-1)),void>::type 
   parallel_execution_native::create_flow(
          InQueue && split_queue,  OutQueue && join_queue,
          SplitJoin<Policy,Transformers...> && split_obj,std::vector<std::thread> & tasks) const
{  
   using namespace std;
   using namespace experimental;
   
   tasks.emplace_back([&](){
     /*auto consumer = [&](typename std::decay<OutQueue>::type::value_type::first_type item){
       join_queue.push( item, index);
     };*/
     auto joiner_tuple = std::tuple<OutQueue &, int> (join_queue, index);
     split_consumer_queue<InQueue> input_queue(split_queue,index);
     do_pipeline(input_queue, split_obj.template flow<index>(), joiner_tuple);
    // join_queue.push( typename std::decay<OutQueue>::type::value_type::first_type{},index);
   });

}

template <std::size_t index, typename InQueue, typename OutQueue, 
          typename ...Transformers, typename Policy,
          template <typename P, typename ... T> class SplitJoin,
          requires_split_join< SplitJoin <Policy, Transformers...>> = 0>
typename std::enable_if<(index != (sizeof...(Transformers)-1)),void>::type 
   parallel_execution_native::create_flow( 
          InQueue && split_queue,  OutQueue && join_queue, 
          SplitJoin<Policy,Transformers...> && split_obj, std::vector<std::thread> & tasks) const
{
   using namespace std;
   using namespace experimental;

   tasks.emplace_back([&](){
    /* auto consumer = [&](typename std::decay<OutQueue>::type::value_type::first_type item){
       join_queue.push( item, index);
     };*/
     auto joiner_tuple = std::tuple<OutQueue &, int> (join_queue, index);
     split_consumer_queue<InQueue> input_queue(split_queue,index);
     do_pipeline(input_queue, split_obj.template flow<index>(), joiner_tuple);
   //  join_queue.push(typename std::decay<OutQueue>::type::value_type::first_type{}, index);
   });
   create_flow<index+1>(split_queue, join_queue, std::forward<SplitJoin<Policy,Transformers...>>(split_obj),tasks);
}

template <typename Queue, typename Policy, typename ... Transformers,
          template <typename P> class Window,
          typename ... OtherTransformers,
          requires_window< Window <Policy>> = 0 >
void parallel_execution_native::do_pipeline(
    Queue && input_queue,
    Window<Policy> && win_obj,
    OtherTransformers && ... other_transform_ops) const
{
    using namespace std;
    using namespace experimental;
   
/*    using window_type = typename std::result_of<decltype(&Policy::get_window)(Policy)>::type;
    using window_optional_type = std::experimental::optional<window_type>;
    using value_type = std::pair <window_optional_type, long> ;

    auto window_queue = make_queue<value_type>();
    std::thread windower([&](){
       auto win =win_obj.get_window();
       auto item = input_queue.pop();
       int order = 0;
       while(item.first){
         if(win.add_item(std::move(*item.first))){
           window_queue.push(make_pair(make_optional(win.get_window()), order));
         }
         item = input_queue.pop();
       }
       window_queue.push(make_pair(window_optional_type{},-1));

    });
 */
    windower_queue<Queue,Policy> window_queue{input_queue, win_obj.get_window()};
    do_pipeline(window_queue, std::forward<OtherTransformers>(other_transform_ops)...);
//    windower.join();
}

template <typename Queue, typename Policy, typename ... Transformers,
          template <typename P> class Window,
          typename ... OtherTransformers,
          requires_active_window< Window <Policy>> = 0 >
void parallel_execution_native::do_pipeline(
    Queue && input_queue,
    Window<Policy> && win_obj,
    OtherTransformers && ... other_transform_ops) const
{
    using namespace std;
    using namespace experimental;

    using window_type = typename std::result_of<decltype(&Policy::get_window)(Policy)>::type;
    using window_optional_type = std::experimental::optional<window_type>;
    using value_type = std::pair <window_optional_type, long> ;
    auto window_queue = make_queue<value_type>();
    std::thread windower([&](){
       auto win = win_obj.get_window();
       auto item = input_queue.pop();
       long order = 0;
       while(item.first){
         if(win.add_item(std::move(*item.first))){
           window_queue.push(make_pair(make_optional(win.get_window()), order));
           order++;
         } 
         item = input_queue.pop();
       }
       window_queue.push(make_pair(window_optional_type{},-1));
    });
 
//    windower_queue<Queue,Policy> window_queue{input_queue, win_obj.get_window()};
    do_pipeline(window_queue, std::forward<OtherTransformers>(other_transform_ops)...);
    windower.join();
}
                           






template <typename Queue, typename Policy, typename ... Transformers,
          template <typename P, typename ... T> class SplitJoin,
          typename ... OtherTransformers,
          requires_split_join< SplitJoin <Policy, Transformers...>> = 0 >
void parallel_execution_native::do_pipeline(
    Queue && input_queue,
    SplitJoin<Policy,Transformers...> && split_obj,
    OtherTransformers && ... other_transform_ops) const
{
    using namespace std;
    using namespace experimental;
    using output_type = typename next_input_type<OtherTransformers...>::type;
    using output_optional_type = experimental::optional<output_type>;
    using output_item_type = pair <output_optional_type, long> ;

    auto split_queue = make_split_queue( input_queue, split_obj.num_transformers(), split_obj.get_policy() );
    joiner_queue<output_item_type> join_queue{split_obj.num_transformers(), queue_size_, queue_mode_};
    
    std::vector<std::thread> tasks{};
    create_flow<0>(split_queue, join_queue, std::forward<SplitJoin<Policy,Transformers...>>(split_obj), tasks);
    
 
    do_pipeline(join_queue, std::forward<OtherTransformers>(other_transform_ops)...);
    std::for_each(tasks.begin(), tasks.end(), [](std::thread & thr ){thr.join();});
}

template <typename Queue, typename Predicate, 
          template <typename> class Filter,
          typename ... OtherTransformers,
          requires_filter<Filter<Predicate>> =0>
void parallel_execution_native::do_pipeline(
    Queue & input_queue, 
    Filter<Predicate> && filter_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
  using namespace experimental;

  using input_item_type = typename Queue::value_type;
  using input_value_type = typename input_item_type::first_type;
  auto filter_queue = make_queue<input_item_type>();

  auto filter_task = [&,this]() {
    auto manager = thread_manager();
    auto item{input_queue.pop()};
    while (item.first) {
      if (filter_obj(*item.first)) {
        filter_queue.push(item);
      }
      else {
        filter_queue.push(make_pair(input_value_type{}, item.second));
      }
      item = input_queue.pop();
    }
    filter_queue.push(make_pair(input_value_type{}, -1));
  };
  thread filter_thread{filter_task};

  auto output_queue = make_queue<input_item_type>();
  thread ordering_thread;
  if (is_ordered()) {
    auto ordering_task = [&]() {
      auto manager = thread_manager();
      vector<input_item_type> elements;
      int current = 0;
      long order = 0;
      auto item{filter_queue.pop()};
      for (;;) {
        if(!item.first && item.second == -1) break; 
        if (item.second == current) {
          if (item.first) {
            output_queue.push(make_pair(item.first,order));
            order++;
          }
          current++;
        }
        else {
          elements.push_back(item);
        }
        // TODO: Probably find_if() + erase 
        for (auto it=elements.begin(); it<elements.end(); it++) {
          if (it->second == current) {
            if (it->first) {
              output_queue.push(make_pair(it->first,order));
              order++;
            }
            elements.erase(it);
            current++;
            break;
          }
        }
        item = filter_queue.pop();
      }
      while (elements.size()>0) {
        // TODO: Probably find_if() + erase 
        for (auto it=elements.begin(); it<elements.end(); it++) {
          if (it->second == current) {
            if(it->first) { 
              output_queue.push(make_pair(it->first,order));
              order++;
            }
            elements.erase(it);
            current++;
            break;
          }
        }
      }
      output_queue.push(item);
    };

    ordering_thread = thread{ordering_task};
    do_pipeline(output_queue, forward<OtherTransformers>(other_transform_ops)...);
    filter_thread.join();
    ordering_thread.join();
  }
  else {
    do_pipeline(filter_queue, forward<OtherTransformers>(other_transform_ops)...);
    filter_thread.join();
  }
}

template <typename Queue, typename Combiner, typename Identity,
          template <typename C, typename I> class Reduce,
          typename ... OtherTransformers,
          requires_reduce<Reduce<Combiner,Identity>> = 0>
void parallel_execution_native::do_pipeline(
    Queue && input_queue, 
    Reduce<Combiner,Identity> && reduce_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
  using namespace experimental;

  using input_item_type = typename decay_t<Queue>::value_type;
  using input_item_value_type = typename input_item_type::first_type::value_type;
  using output_item_value_type = optional<decay_t<Identity>>;
  using output_item_type = pair<output_item_value_type,long>;
  auto output_queue = make_queue<output_item_type>();

  auto reduce_task = [&,this]() {
    auto manager = thread_manager();
    auto item{input_queue.pop()};
    int order = 0;
    while (item.first) {
      reduce_obj.add_item(std::forward<Identity>(*item.first));
      item = input_queue.pop();
      if (reduce_obj.reduction_needed()) {
        constexpr sequential_execution seq;
        auto red = reduce_obj.reduce_window(seq);
        output_queue.push(make_pair(red, order++));
      }
    }
    output_queue.push(make_pair(output_item_value_type{}, -1));
  };
  thread reduce_thread{reduce_task};
  do_pipeline(output_queue, forward<OtherTransformers>(other_transform_ops)...);
  reduce_thread.join();
}

template <typename Queue, typename Transformer, typename Predicate,
          template <typename T, typename P> class Iteration,
          typename ... OtherTransformers,
          requires_iteration<Iteration<Transformer,Predicate>> =0,
          requires_no_pattern<Transformer> =0>
void parallel_execution_native::do_pipeline(
    Queue & input_queue, 
    Iteration<Transformer,Predicate> && iteration_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
  using namespace experimental;

  using input_item_type = typename decay_t<Queue>::value_type;
  using input_item_value_type = typename input_item_type::first_type::value_type;
  auto output_queue = make_queue<input_item_type>();

  auto iteration_task = [&]() {
    for (;;) {
      auto item = input_queue.pop();
      if (!item.first) break;
      auto value = iteration_obj.transform(*item.first);
      auto new_item = input_item_type{value,item.second};
      if (iteration_obj.predicate(value)) {
        output_queue.push(new_item);
      }
      else {
        input_queue.push(new_item);
      }
    }
    while (!input_queue.is_empty()) {
      auto item = input_queue.pop();
      auto value = iteration_obj.transform(*item.first);
      auto new_item = input_item_type{value,item.second};
      if (iteration_obj.predicate(value)) {
        output_queue.push(new_item);
      }
      else {
        input_queue.push(new_item);
      }
    }
    output_queue.push(input_item_type{{},-1});
  };

  thread iteration_thread{iteration_task};
  do_pipeline(output_queue, forward<OtherTransformers>(other_transform_ops)...);
  iteration_thread.join();
}

template <typename Queue, typename Transformer, typename Predicate,
          template <typename T, typename P> class Iteration,
          typename ... OtherTransformers,
          requires_iteration<Iteration<Transformer,Predicate>> =0,
          requires_pipeline<Transformer> =0>
void parallel_execution_native::do_pipeline(
    Queue & input_queue, 
    Iteration<Transformer,Predicate> && iteration_obj,
    OtherTransformers && ... other_transform_ops) const
{
  static_assert(!is_pipeline<Transformer>, "Not implemented");
}


template <typename Queue, typename ... Transformers,
          template <typename...> class Pipeline,
          requires_pipeline<Pipeline<Transformers...>> = 0>
void parallel_execution_native::do_pipeline(
    Queue & input_queue,
    Pipeline<Transformers...> && pipeline_obj) const
{
  do_pipeline_nested(
      input_queue,
      pipeline_obj.transformers(), 
      std::make_index_sequence<sizeof...(Transformers)>());
}

template <typename Queue, typename ... Transformers,
          template <typename...> class Pipeline,
          typename ... OtherTransformers,
          requires_pipeline<Pipeline<Transformers...>> = 0>
void parallel_execution_native::do_pipeline(
    Queue & input_queue,
    Pipeline<Transformers...> && pipeline_obj,
    OtherTransformers && ... other_transform_ops) const
{
  do_pipeline_nested(
      input_queue,
      std::tuple_cat(pipeline_obj.transformers(), 
          std::forward_as_tuple(other_transform_ops...)),
      std::make_index_sequence<sizeof...(Transformers) + sizeof...(OtherTransformers)>());
}

template <typename Queue, typename ... Transformers,
          std::size_t ... I>
void parallel_execution_native::do_pipeline_nested(
    Queue & input_queue, 
    std::tuple<Transformers...> && transform_ops,
    std::index_sequence<I...>) const
{
  do_pipeline(input_queue,
      std::forward<Transformers>(std::get<I>(transform_ops))...);
}

} // end namespace grppi

#endif
