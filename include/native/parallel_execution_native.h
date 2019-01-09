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
#ifndef GRPPI_NATIVE_PARALLEL_EXECUTION_NATIVE_H
#define GRPPI_NATIVE_PARALLEL_EXECUTION_NATIVE_H

#include "worker_pool.h"
#include "../common/mpmc_queue.h"
#include "../common/iterator.h"
#include "../common/execution_traits.h"
#include "../common/configuration.h"

#include <thread>
#include <atomic>
#include <algorithm>
#include <vector>
#include <type_traits>
#include <tuple>
#if __cplusplus < 201703L
#include <experimental/optional>
#else
#include <optional>
#endif
#include <sstream>
#include <cstdlib>
#include <cstring>

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
  thread_registry() = default;

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
  std::vector<std::thread::id> ids_{};
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
}

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
  parallel_execution_native() noexcept  
  {}

  parallel_execution_native(int concurrency_degree) noexcept :
    concurrency_degree_{concurrency_degree}
  {}

  /** 
  \brief Constructs a native parallel execution policy.

  Creates a parallel execution native object selecting the concurrency degree
  and ordering mode.
  \param concurrency_degree Number of threads used for parallel algorithms.
  \param order Whether ordered executions is enabled or disabled.
  */
  parallel_execution_native(int concurrency_degree, bool ordering) noexcept :
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
  \brief Get number of grppi threads.
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
  [[deprecated("Thread ids are deprecated.\n"
               "If you have a specific use case file a bug")]]
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
  \brief Returns the reference of a communication queue for elements of type T 
  if the queue has been created in an outer pattern.
  Returns the reference of the queue received as argument.
  \tparam T Element type for the queue.
  \tparam Transformers List of the next transformers.
  \param queue Reference of a queue of type T
  */
  template <typename T, typename ... Transformers>
  mpmc_queue<T>& get_output_queue(mpmc_queue<T> & queue, Transformers && ...) const {
    return queue;
  }

  /**
  \brief Makes a communication queue for elements of type T 
  if the queue has not been created in an outer pattern.
  Call to the make_queue function and the value is returned via move semantics.
  \tparam T Element type for the queue.
  \tparam Transformers List of the next transformers.
  */
  template <typename T, typename ... Transformers>
  mpmc_queue<T> get_output_queue(Transformers && ...) const{
    return std::move(make_queue<T>());
  }

  /**
  \brief Applies a transformation to multiple sequences leaving the result in
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
  [[deprecated("Use new interface with predicate argument")]]
  auto divide_conquer(Input && input, 
                      Divider && divide_op, 
                      Solver && solve_op, 
                      Combiner && combine_op) const; 

  /**
  \brief Invoke \ref md_divide-conquer.
  \tparam Input Type used for the input problem.
  \tparam Divider Callable type for the divider operation.
  \tparam Predicate Callable type for the stop condition predicate.
  \tparam Solver Callable type for the solver operation.
  \tparam Combiner Callable type for the combiner operation.
  \param ex Sequential execution policy object.
  \param input Input problem to be solved.
  \param divider_op Divider operation.
  \param predicate_op Predicate operation.
  \param solver_op Solver operation.
  \param combine_op Combiner operation.
  */
  template <typename Input, typename Divider, typename Predicate, typename Solver, typename Combiner>
  auto divide_conquer(Input && input,
                      Divider && divide_op,
                      Predicate && predicate_op,
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
  \brief Invoke \ref md_pipeline coming from another context
  that uses mpmc_queues as communication channels.
  \tparam InputType Type of the input stream.
  \tparam Transformers Callable types for the transformers in the pipeline.
  \tparam InputType Type of the output stream.
  \param input_queue Input stream communicator.
  \param transform_ops Transformer operations.
  \param output_queue Input stream communicator.
  */
  template <typename InputType, typename Transformer, typename OutputType>
  void pipeline(mpmc_queue<InputType> & input_queue, Transformer && transform_op,
                mpmc_queue<OutputType> &output_queue) const
  {
    do_pipeline(input_queue, std::forward<Transformer>(transform_op), output_queue);
  }

private:

  template <typename Input, typename Divider, typename Solver, typename Combiner>
  auto divide_conquer(Input && input, 
                      Divider && divide_op, 
                      Solver && solve_op, 
                      Combiner && combine_op,
                      std::atomic<int> & num_threads) const; 

 template <typename Input, typename Divider,typename Predicate, typename Solver, typename Combiner>
  auto divide_conquer(Input && input,
                      Divider && divide_op,
                      Predicate && predicate_op,
                      Solver && solve_op,
                      Combiner && combine_op,
                      std::atomic<int> & num_threads) const;


  template <typename Queue, typename Consumer,
            requires_no_pattern<Consumer> = 0>
  void do_pipeline(Queue & input_queue, Consumer && consume_op) const;

  template <typename Inqueue, typename Transformer, typename output_type,
            requires_no_pattern<Transformer> = 0>
  void do_pipeline(Inqueue & input_queue, Transformer && transform_op, 
      mpmc_queue<output_type> & output_queue) const;

  template <typename T, typename ... Others>
  void do_pipeline(mpmc_queue<T> & in_q, mpmc_queue<T> & same_queue, Others &&... ops) const;
   
  template <typename T>
  void do_pipeline(mpmc_queue<T> &) const {}


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

  template <typename Queue, typename Execution, typename Transformer, 
            template <typename, typename> class Context,
            typename ... OtherTransformers,
            requires_context<Context<Execution,Transformer>> = 0>
  void do_pipeline(Queue & input_queue, Context<Execution,Transformer> && context_op, 
       OtherTransformers &&... other_ops) const;

  template <typename Queue, typename Execution, typename Transformer, 
            template <typename, typename> class Context,
            typename ... OtherTransformers,
            requires_context<Context<Execution,Transformer>> = 0>
  void do_pipeline(Queue & input_queue, Context<Execution,Transformer> & context_op, 
       OtherTransformers &&... other_ops) const
  {
    do_pipeline(input_queue, std::move(context_op),
      std::forward<OtherTransformers>(other_ops)...);
  }

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
  }

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

  mutable thread_registry thread_registry_{};
  
  configuration<> config_{};

  int concurrency_degree_ = config_.concurrency_degree();
  
  bool ordering_ = config_.ordering();
  
  int queue_size_ = config_.queue_size();

  queue_mode queue_mode_ = config_.mode();
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
        std::forward<Identity>(identity),
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

  return seq.reduce(partial_results.begin(), 
     partial_results.size(), std::forward<Identity>(identity),
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


template <typename Input, typename Divider,typename Predicate, typename Solver, typename Combiner>
auto parallel_execution_native::divide_conquer(
    Input && problem,
    Divider && divide_op,
    Predicate && predicate_op,
    Solver && solve_op,
    Combiner && combine_op) const
{
  std::atomic<int> num_threads{concurrency_degree_-1};

  return divide_conquer(std::forward<Input>(problem), std::forward<Divider>(divide_op),
        std::forward<Predicate>(predicate_op),
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

template <typename Input, typename Divider,typename Predicate, typename Solver, typename Combiner>
auto parallel_execution_native::divide_conquer(
    Input && input,
    Divider && divide_op,
    Predicate && predicate_op,
    Solver && solve_op,
    Combiner && combine_op,
    std::atomic<int> & num_threads) const
{
  constexpr sequential_execution seq;
  if (num_threads.load() <=0) {
    return seq.divide_conquer(std::forward<Input>(input),
        std::forward<Divider>(divide_op),
        std::forward<Predicate>(predicate_op),
        std::forward<Solver>(solve_op),
        std::forward<Combiner>(combine_op));
  }

  if (predicate_op(input)) { return solve_op(std::forward<Input>(input)); }
  auto subproblems = divide_op(std::forward<Input>(input));

  using subresult_type =
      std::decay_t<typename std::result_of<Solver(Input)>::type>;
  std::vector<subresult_type> partials(subproblems.size()-1);

  auto process_subproblem = [&,this](auto it, std::size_t div) {
    partials[div] = this->divide_conquer(std::forward<Input>(*it),
        std::forward<Divider>(divide_op), std::forward<Predicate>(predicate_op),
        std::forward<Solver>(solve_op),
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
        std::forward<Divider>(divide_op), std::forward<Predicate>(predicate_op), std::forward<Solver>(solve_op),
        std::forward<Combiner>(combine_op));
  }

  auto subresult = divide_conquer(std::forward<Input>(*subproblems.begin()),
      std::forward<Divider>(divide_op), std::forward<Predicate>(predicate_op), std::forward<Solver>(solve_op),
      std::forward<Combiner>(combine_op), num_threads);

  workers.wait();

  return seq.reduce(partials.begin(), partials.size(),
      std::forward<subresult_type>(subresult), std::forward<Combiner>(combine_op));
}
template <typename Queue, typename Consumer,
          requires_no_pattern<Consumer>>
void parallel_execution_native::do_pipeline(
    Queue & input_queue, 
    Consumer && consume_op) const
{
  using namespace std;
  using input_type = typename Queue::value_type;

  auto manager = thread_manager();
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
    auto it = find_if(elements.begin(), elements.end(), 
       [&](auto x) { return x.second== current; });
    if(it != elements.end()){
      consume_op(*it->first);
      elements.erase(it);
      current++;
    }
  }
  while (elements.size()>0) {
    auto it = find_if(elements.begin(), elements.end(), 
       [&](auto x) { return x.second== current; });
    if(it != elements.end()){
      consume_op(*it->first);
      elements.erase(it);
      current++;
    }
  }
}


template <typename Inqueue, typename Transformer, typename output_type,
            requires_no_pattern<Transformer>>
void parallel_execution_native::do_pipeline(Inqueue & input_queue, Transformer && transform_op,
      mpmc_queue<output_type> & output_queue) const
{
  using namespace std;
#if __cplusplus < 201703L
  using namespace experimental;
#endif

  using output_item_value_type = typename output_type::first_type::value_type;
  for (;;) {
    auto item{input_queue.pop()}; 
    if(!item.first) break;
    auto out = output_item_value_type{transform_op(*item.first)};
    output_queue.push(make_pair(out,item.second)) ;
  }
}



template <typename Queue, typename Transformer, 
          typename ... OtherTransformers,
          requires_no_pattern<Transformer>>
void parallel_execution_native::do_pipeline(
    Queue & input_queue, 
    Transformer && transform_op,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
#if __cplusplus < 201703L
  using namespace experimental;
#endif

  using input_item_type = typename Queue::value_type;
  using input_item_value_type = typename input_item_type::first_type::value_type;
  using transform_result_type = 
      decay_t<typename result_of<Transformer(input_item_value_type)>::type>;
  using output_item_value_type = optional<transform_result_type>;
  using output_item_type = pair<output_item_value_type,long>;

  decltype(auto) output_queue =
    get_output_queue<output_item_type>(other_transform_ops...);

  thread task([&,this]() {
    auto manager = thread_manager();

    for (;;) {
      auto item{input_queue.pop()};
      if (!item.first) break;
      auto out = output_item_value_type{transform_op(*item.first)};
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
          requires_farm<Farm<FarmTransformer>>>
void parallel_execution_native::do_pipeline(
    Queue & input_queue, 
    Farm<FarmTransformer> && farm_obj) const
{
  using namespace std;

  auto farm_task = [&](int) {
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


template <typename Queue, typename Execution, typename Transformer,
          template <typename, typename> class Context,
          typename ... OtherTransformers,
          requires_context<Context<Execution,Transformer>>>
void parallel_execution_native::do_pipeline(Queue & input_queue, 
    Context<Execution,Transformer> && context_op,
    OtherTransformers &&... other_ops) const 
{
  using namespace std;
#if __cplusplus < 201703L
  using namespace experimental;
#endif

  using input_item_type = typename Queue::value_type;
  using input_item_value_type = typename input_item_type::first_type::value_type;

  using output_type = typename stage_return_type<input_item_value_type, Transformer>::type;
  using output_optional_type = optional<output_type>;
  using output_item_type = pair <output_optional_type, long> ;

  decltype(auto) output_queue =
    get_output_queue<output_item_type>(other_ops...);
  
  auto context_task = [&]() {
    context_op.execution_policy().pipeline(input_queue, context_op.transformer(), output_queue);
    output_queue.push( make_pair(output_optional_type{}, -1) );
  };

  worker_pool workers{1};
  workers.launch_tasks(*this, context_task);

  do_pipeline(output_queue,
      forward<OtherTransformers>(other_ops)... );

  workers.wait();
}


template <typename Queue, typename FarmTransformer, 
          template <typename> class Farm,
          typename ... OtherTransformers,
          requires_farm<Farm<FarmTransformer>>>
void parallel_execution_native::do_pipeline(
    Queue & input_queue, 
    Farm<FarmTransformer> && farm_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
#if __cplusplus < 201703L
  using namespace experimental;
#endif

  using input_item_type = typename Queue::value_type;
  using input_item_value_type = typename input_item_type::first_type::value_type;

  using output_type = typename stage_return_type<input_item_value_type, FarmTransformer>::type;
  using output_optional_type = optional<output_type>;
  using output_item_type = pair <output_optional_type, long> ;

  decltype(auto) output_queue = 
    get_output_queue<output_item_type>(other_transform_ops...);

  atomic<int> done_threads{0};

  auto farm_task = [&](int nt) {
    do_pipeline(input_queue, farm_obj.transformer(), output_queue);
    done_threads++;
    if (done_threads == nt) {
      output_queue.push(make_pair(output_optional_type{}, -1));
    }else{
      input_queue.push(input_item_type{});
    }
  };

  auto ntasks = farm_obj.cardinality();
  worker_pool workers{ntasks};
  workers.launch_tasks(*this, farm_task, ntasks);  
  do_pipeline(output_queue, 
      forward<OtherTransformers>(other_transform_ops)... );
  
  workers.wait();
}

template <typename Queue, typename Predicate, 
          template <typename> class Filter,
          typename ... OtherTransformers,
          requires_filter<Filter<Predicate>>>
void parallel_execution_native::do_pipeline(
    Queue & input_queue, 
    Filter<Predicate> && filter_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
#if __cplusplus < 201703L
  using namespace experimental;
#endif

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

  decltype(auto) output_queue =
    get_output_queue<input_item_type>(other_transform_ops...);

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
        auto it = find_if(elements.begin(), elements.end(), 
           [&](auto x) { return x.second== current; });
        if(it != elements.end()){
          if (it->first) {
            output_queue.push(make_pair(it->first,order));
            order++;
          }       
          elements.erase(it);
          current++;
        }
        item = filter_queue.pop();
      }
      while (elements.size()>0) {
        auto it = find_if(elements.begin(), elements.end(), 
           [&](auto x) { return x.second== current; });
        if(it != elements.end()){
          if (it->first) {
            output_queue.push(make_pair(it->first,order));
            order++;
          }       
          elements.erase(it);
          current++;
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
          requires_reduce<Reduce<Combiner,Identity>>>
void parallel_execution_native::do_pipeline(
    Queue && input_queue, 
    Reduce<Combiner,Identity> && reduce_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
#if __cplusplus < 201703L
  using namespace experimental;
#endif

  using output_item_value_type = optional<decay_t<Identity>>;
  using output_item_type = pair<output_item_value_type,long>;
  decltype(auto) output_queue =
    get_output_queue<output_item_type>(other_transform_ops...);

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
          requires_iteration<Iteration<Transformer,Predicate>>,
          requires_no_pattern<Transformer>>
void parallel_execution_native::do_pipeline(
    Queue & input_queue, 
    Iteration<Transformer,Predicate> && iteration_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
#if __cplusplus < 201703L
  using namespace experimental;
#endif

  using input_item_type = typename decay_t<Queue>::value_type;

  decltype(auto) output_queue =
    get_output_queue<input_item_type>(other_transform_ops...);

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
    while (!input_queue.empty()) {
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
          requires_iteration<Iteration<Transformer,Predicate>>,
          requires_pipeline<Transformer>>
void parallel_execution_native::do_pipeline(
    Queue &,
    Iteration<Transformer,Predicate> &&,
    OtherTransformers && ...) const
{
  static_assert(!is_pipeline<Transformer>, "Not implemented");
}


template <typename Queue, typename ... Transformers,
          template <typename...> class Pipeline,
          requires_pipeline<Pipeline<Transformers...>>>
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
          requires_pipeline<Pipeline<Transformers...>>>
void parallel_execution_native::do_pipeline(
    Queue & input_queue,
    Pipeline<Transformers...> && pipeline_obj,
    OtherTransformers && ... other_transform_ops) const
{
  do_pipeline_nested(
      input_queue,
      std::tuple_cat(pipeline_obj.transformers(), 
          std::forward_as_tuple(other_transform_ops...)),
      std::make_index_sequence<sizeof...(Transformers)+sizeof...(OtherTransformers)>());
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

template<typename T, typename... Others>
void parallel_execution_native::do_pipeline(mpmc_queue<T> &, mpmc_queue<T> &, Others &&...) const
{
}

} // end namespace grppi

#endif
