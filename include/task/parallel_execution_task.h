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
#ifndef GRPPI_NATIVE_PARALLEL_EXECUTION_TASK_H
#define GRPPI_NATIVE_PARALLEL_EXECUTION_TASK_H

#include "pool.h"
#include "../common/mpmc_queue.h"
#include "../common/iterator.h"
#include "../common/execution_traits.h"
#include "../common/configuration.h"

#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include <vector>
#include <type_traits>
#include <tuple>
#include <experimental/optional>
#include <sstream>
#include <cstdlib>
#include <cstring>

namespace grppi {


class execution_task_base
{};


/** 
 \brief Native task-based parallel execution policy.
 This policy uses ISO C++ threads as implementation building block allowing
 usage in any ISO C++ compliant platform.
 */
template <typename Scheduler>
class parallel_execution_task : public execution_task_base {
public:
  // Alias type for the scheduler and the task type
  using scheduler_type = Scheduler;
  using task_type = typename Scheduler::task_type;

  /** 
  \brief Default construct a task parallel execution policy.

  Creates a parallel execution task object.

  The concurrency degree is determined by the platform.

  \note The concurrency degree is fixed to 2 times the hardware concurrency
   degree.
  */
  parallel_execution_task(Scheduler &s) noexcept :
   scheduler{s},thread_pool{scheduler,concurrency_degree_}
  {
  }

  parallel_execution_task(Scheduler &s, int concurrency_degree) noexcept :
    concurrency_degree_{concurrency_degree},
    scheduler{s}, thread_pool{scheduler,concurrency_degree_}
  {
  }

  /** 
  \brief Constructs a task parallel execution policy.

  Creates a parallel execution task object selecting the concurrency degree
  and ordering mode.
  \param concurrency_degree Number of threads used for parallel algorithms.
  \param order Whether ordered executions is enabled or disabled.
  */
  parallel_execution_task(Scheduler s, int concurrency_degree, bool ordering) noexcept :
    concurrency_degree_{concurrency_degree},
    ordering_{ordering},
    scheduler{s}, thread_pool{scheduler,concurrency_degree_}
  {
  }

  parallel_execution_task(const parallel_execution_task & ex) :
      parallel_execution_task{ex.scheduler, ex.concurrency_degree_, ex.ordering_}
  {}

   
  /** 
  \brief Destroy a task parallel execution policy.

  */
  ~parallel_execution_task(){
     thread_pool.finalize_pool();
  }

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
  std::shared_ptr<mpmc_queue<T>> make_queue() const {
    return std::make_shared<mpmc_queue<T>>(mpmc_queue<T>(queue_size_, queue_mode_));
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
  std::shared_ptr<mpmc_queue<T>> get_output_queue(std::shared_ptr<mpmc_queue<T>> & queue, Transformers && ...) const {
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
  std::shared_ptr<mpmc_queue<T>> get_output_queue(Transformers && ...) const{
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
  void pipeline(std::shared_ptr<mpmc_queue<InputType>> & input_queue, Transformer && transform_op,
                std::shared_ptr<mpmc_queue<OutputType>> &output_queue) const
  {
    do_pipeline(input_queue, std::forward<Transformer>(transform_op), output_queue);
  }

private:

  /*template <typename Input, typename Divider, typename Solver, typename Combiner>
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
*/

  template <typename Queue, typename Consumer,
            requires_no_pattern<Consumer> = 0>
  void do_pipeline(Queue & input_queue, Consumer && consume_op) const;

  template <typename Inqueue, typename Transformer, typename output_type,
            requires_no_pattern<Transformer> = 0>
  void do_pipeline(Inqueue & input_queue, Transformer && transform_op, 
      std::shared_ptr<mpmc_queue<output_type>> & output_queue) const;

  template <typename T, typename ... Others>
  void do_pipeline(std::shared_ptr<mpmc_queue<T>> & in_q, std::shared_ptr<mpmc_queue<T>> & same_queue, Others &&... ops) const;
   
  template <typename T>
  void do_pipeline(std::shared_ptr<mpmc_queue<T>> &) const{}


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

  mutable Scheduler scheduler;  

  configuration<> config_{};

  int concurrency_degree_ = config_.concurrency_degree();
  
  mutable pool<Scheduler> thread_pool;
  
  bool ordering_ = config_.ordering();
  
  int queue_size_ = config_.queue_size();

  queue_mode queue_mode_ = config_.mode();
};

/**
\brief Metafunction that determines if type E is parallel_execution_task
\tparam Execution policy type.
*/
template <typename E,typename T>
constexpr bool is_parallel_execution_task() {
  return std::is_same<E, parallel_execution_task<T> >::value;
}

template <typename T>
using execution_task = parallel_execution_task<T>;

/**
\brief Determines if an execution policy is supported in the current compilation.
\note Specialization for parallel_execution_task.
*/
template <typename U>
struct support<parallel_execution_task<U>>
{
   static constexpr bool value = true;
};

/*template <>
constexpr bool is_supported<execution_task>() { return true; }
*/
/**
\brief Determines if an execution policy supports the map pattern.
\note Specialization for parallel_execution_task.
*/
/*template <>
constexpr bool supports_map<execution_task>() { return true; }
*/
/**
\brief Determines if an execution policy supports the reduce pattern.
\note Specialization for parallel_execution_task.
*/
/*template <>
constexpr bool supports_reduce<parallel_execution_task>() { return true; }
*/
/**
\brief Determines if an execution policy supports the map-reduce pattern.
\note Specialization for parallel_execution_task.
*/
/*template <>
constexpr bool supports_map_reduce<parallel_execution_task>() { return true; }
*/
/**
\brief Determines if an execution policy supports the stencil pattern.
\note Specialization for parallel_execution_task.
*/
/*template <>
constexpr bool supports_stencil<parallel_execution_task>() { return true; }
*/
/**
\brief Determines if an execution policy supports the divide/conquer pattern.
\note Specialization for parallel_execution_task.
*/
/*template <>
constexpr bool supports_divide_conquer<parallel_execution_task>() { return true; }
*/
/**
\brief Determines if an execution policy supports the pipeline pattern.
\note Specialization for parallel_execution_task.
*/
/*template <>
constexpr bool supports_pipeline<parallel_execution_task>() { return true; }
*/
template <typename Scheduler>
template <typename ... InputIterators, typename OutputIterator, 
          typename Transformer>
void parallel_execution_task<Scheduler>::map(
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


  //Naive implementation of the data manager
  std::atomic_flag used[100] = {ATOMIC_FLAG_INIT};
  std::tuple < std::tuple<InputIterators...>, std::size_t, OutputIterator > data_manager[100];
  
  // Define granularity
  int chunk_size = sequence_size / concurrency_degree_;
  if(chunk_size == 0) chunk_size = sequence_size;
  //int chunk_size = 48;

  int function_id = scheduler.register_data_parallel_task([this,&process_chunk,&data_manager,&used](task_type &task){
    //Get data
    auto input_data = data_manager[task.get_data_location()];
    process_chunk(std::get<0>(input_data), std::get<1>(input_data), std::get<2>(input_data));
    //Notify consume and free the position
    used[task.get_data_location()].clear();
    scheduler.notify_consume();
  });

  auto aux_in = firsts;
  auto aux_out = first_out;

  auto p_id = scheduler.start_pattern();
  for(unsigned int i=0; i< sequence_size / chunk_size; ++i){
    bool introduced=false;
    int pos = 0;
    //Check for an empty position
    while(!introduced){
      for(int j = 0; j<100; j++){
        if(!used[j].test_and_set()) {
          introduced=true;
          pos = j;
        }
      }
    }

    // Last iteration
    if( i == (sequence_size/chunk_size)-1 ) 
      chunk_size += sequence_size % chunk_size;

    //Store data in data manager
    data_manager[pos] = make_tuple(aux_in, chunk_size, aux_out);
    task_type t{function_id, scheduler.get_task_id()};
    t.set_data_location(pos);
    t.set_pattern_id(p_id);
    //Launch the task
    scheduler.launch_data_task(t);
    aux_in = iterators_next(aux_in, chunk_size);
    aux_out = next(aux_out, chunk_size);
  }

  scheduler.run_data(p_id);

}

template <typename Scheduler>
template <typename InputIterator, typename Identity, typename Combiner>
auto parallel_execution_task<Scheduler>::reduce(
    InputIterator first, std::size_t sequence_size,
    Identity && identity,
    Combiner && combine_op) const
{
  using result_type = std::decay_t<Identity>;
 
  // Define granularity
  int chunk_size = sequence_size / concurrency_degree_;
  if(chunk_size == 0) chunk_size = sequence_size;

  std::vector<result_type> partial_results( sequence_size/chunk_size );   

  constexpr sequential_execution seq;
  auto process_chunk = [&](InputIterator f, std::size_t sz, std::size_t id) {
    partial_results[id] = seq.reduce(f,sz, std::forward<Identity>(identity), 
        std::forward<Combiner>(combine_op));
  };
  
  //Naive implementation of the data manager - Only std cointainers
  std::atomic_flag used[100] = {ATOMIC_FLAG_INIT};
  std::tuple < InputIterator, std::size_t, std::size_t > data_manager[100];

  int function_id = scheduler.register_data_parallel_task([this, &process_chunk, &data_manager, &used](task_type &task){
      auto input_data = data_manager[task.get_data_location()];
      process_chunk(std::get<0>(input_data), std::get<1>(input_data), std::get<2>(input_data));
      //Notify consume and free the position
      used[task.get_data_location()].clear();
      scheduler.notify_consume();
  });

  auto p_id = scheduler.start_pattern();
  auto aux_in = first;

  for(unsigned int i=0; i< sequence_size / chunk_size; ++i){
    bool introduced=false;
    int pos = 0;

    //Check for an empty position
    while(!introduced){
      for(int j = 0; j<100; j++){
        if(!used[j].test_and_set()) {
          introduced=true;
          pos = j;
        }
      }
    }

    // Last iteration
    if( i == (sequence_size/chunk_size)-1 ) 
      chunk_size += sequence_size % chunk_size;

    //Store data in data manager
    data_manager[pos] = make_tuple(aux_in, chunk_size, i);
    task_type t{function_id, scheduler.get_task_id()};
    t.set_data_location(pos);
    t.set_pattern_id(p_id);
    //Launch the task
    scheduler.launch_data_task(t);
    aux_in = std::next(aux_in, chunk_size);
  }

  scheduler.run_data(p_id);

  return seq.reduce(partial_results.begin(), 
      partial_results.size(), std::forward<Identity>(identity), 
      std::forward<Combiner>(combine_op));
}

template <typename Scheduler>
template <typename ... InputIterators, typename Identity, 
          typename Transformer, typename Combiner>
auto parallel_execution_task<Scheduler>::map_reduce(
    std::tuple<InputIterators...> firsts, 
    std::size_t sequence_size,
    Identity &&,
    Transformer && transform_op, Combiner && combine_op) const
{
  using result_type = std::decay_t<Identity>;

  // Define granularity
  int chunk_size = sequence_size / concurrency_degree_;
  if(chunk_size == 0) chunk_size = sequence_size;
  std::vector<result_type> partial_results( sequence_size/chunk_size );   

  constexpr sequential_execution seq;
  auto process_chunk = [&](auto f, std::size_t sz, std::size_t id) {
    partial_results[id] = seq.map_reduce(f, sz,
        std::forward<Identity>(partial_results[id]),
        std::forward<Transformer>(transform_op),
        std::forward<Combiner>(combine_op));
  };

  //Naive implementation of the data manager - Only std cointainers
  std::atomic_flag used[100] = {ATOMIC_FLAG_INIT};
  std::tuple < std::tuple<InputIterators...>, std::size_t, std::size_t > data_manager[100];

  int function_id = scheduler.register_data_parallel_task([this, &process_chunk, &data_manager, &used](task_type &task){
      auto input_data = data_manager[task.get_data_location()];
      process_chunk(std::get<0>(input_data), std::get<1>(input_data), std::get<2>(input_data));
      //Notify consume and free the position
      used[task.get_data_location()].clear();
      scheduler.notify_consume();
  });

  auto aux_in = firsts;
  auto p_id = scheduler.start_pattern();

  for(unsigned int i=0; i< sequence_size / chunk_size; ++i){
    bool introduced=false;
    int pos = 0;
    //Check for an empty position
    while(!introduced){
      for(int j = 0; j<100; j++){
        if(!used[j].test_and_set()) {
          introduced=true;
          pos = j;
        }
      }
    }

    // Last iteration
    if( i == (sequence_size/chunk_size)-1 ) 
      chunk_size += sequence_size % chunk_size;

    //Store data in data manager
    data_manager[pos] = make_tuple(aux_in, chunk_size, i);
    task_type t{function_id, scheduler.get_task_id()};
    t.set_data_location(pos);
    t.set_pattern_id(p_id);
    //Launch the task
    scheduler.launch_data_task(t);
    aux_in = iterators_next(aux_in, chunk_size);
  }

  scheduler.run_data(p_id);

  return seq.reduce(std::next(partial_results.begin()), 
     partial_results.size()-1, std::forward<result_type>(partial_results[0]),
     std::forward<Combiner>(combine_op));
}

template <typename Scheduler>
template <typename ... InputIterators, typename OutputIterator,
          typename StencilTransformer, typename Neighbourhood>
void parallel_execution_task<Scheduler>::stencil(
    std::tuple<InputIterators...> firsts, OutputIterator first_out,
    std::size_t sequence_size,
    StencilTransformer && transform_op,
    Neighbourhood && neighbour_op) const
{
  constexpr sequential_execution seq;

  // Define granularity
  int chunk_size = sequence_size / concurrency_degree_;
  if(chunk_size == 0) chunk_size = sequence_size;

  auto process_chunk =
  [&transform_op, &neighbour_op,seq](auto fins, std::size_t sz, auto fout)
  {
    seq.stencil(fins, fout, sz,
      std::forward<StencilTransformer>(transform_op),
      std::forward<Neighbourhood>(neighbour_op));
  };

  //Naive implementation of the data manager - Only std cointainers
  std::atomic_flag used[100] = {ATOMIC_FLAG_INIT};
  std::tuple < std::tuple<InputIterators...>, std::size_t, OutputIterator > data_manager[100];

  int function_id = scheduler.register_data_parallel_task([this, &process_chunk, &data_manager, &used](task_type &task){
      auto input_data = data_manager[task.get_data_location()];
      process_chunk(std::get<0>(input_data), std::get<1>(input_data), std::get<2>(input_data));
      //Notify consume and free the position
      used[task.get_data_location()].clear();
      scheduler.notify_consume();
  });

  auto aux_in = firsts;
  auto aux_out = first_out;
  auto p_id = scheduler.start_pattern();

  for(unsigned int i=0; i< sequence_size / chunk_size; ++i){
    bool introduced=false;
    int pos = 0;
    //Check for an empty position
    while(!introduced){
      for(int j = 0; j<100; j++){
        if(!used[j].test_and_set()) {
          introduced=true;
          pos = j;
        }
      }
    }

    // Last iteration
    if( i == (sequence_size/chunk_size)-1 ) 
      chunk_size += sequence_size % chunk_size;

    //Store data in data manager
    data_manager[pos] = make_tuple(aux_in, chunk_size, aux_out);
    task_type t{function_id, scheduler.get_task_id()};
    t.set_data_location(pos);
    t.set_pattern_id(p_id);
    //Launch the task
    scheduler.launch_data_task(t);
    aux_in = iterators_next(aux_in, chunk_size);
    aux_out = std::next(aux_out, chunk_size);
  }

  scheduler.run_data(p_id);

}


template <typename Scheduler>
template <typename Input, typename Divider, typename Solver, typename Combiner>
auto parallel_execution_task<Scheduler>::divide_conquer(
    Input && problem, 
    Divider && divide_op, 
    Solver && solve_op, 
    Combiner && combine_op) const
{ 
}

template <typename Scheduler>
template <typename Input, typename Divider,typename Predicate, typename Solver, typename Combiner>
auto parallel_execution_task<Scheduler>::divide_conquer(
    Input && input,
    Divider && divide_op,
    Predicate && predicate_op,
    Solver && solve_op,
    Combiner && combine_op) const
{
  using result_type = 
    std::decay_t<typename std::result_of<Solver(Input)>::type>;
  //Naive implementation of the data manager 
  std::atomic_flag used[1000] = {ATOMIC_FLAG_INIT};
  std::tuple < std::decay_t<Input>, result_type > data_manager[1000];
  constexpr sequential_execution seq;

  // Merge task
  int merger_id =scheduler.register_data_parallel_task([&combine_op,&data_manager,&used,this](task_type &task)
  {
    auto data = &data_manager[task.get_data_location()];
    for(auto it = task.multiple_input_location.begin(); it != task.multiple_input_location.end(); it++) {
      auto input = data_manager[*it];
      std::get<1>(*data) = combine_op(std::get<1>(*data), std::get<1>(input));
      used[*it].clear();
    }
    scheduler.notify_consume();
  });

  int function_id = scheduler.register_data_parallel_task(
     [&divide_op,&predicate_op, &solve_op,&combine_op, &data_manager, &used, this, &seq, &merger_id](task_type &task){
       auto data = &data_manager[task.get_data_location()];
       if( predicate_op(std::get<0>(*data) ) ) {
         std::get<1>(*data) = solve_op( std::get<0>(*data) );
         scheduler.notify_consume();
         return;
       }
       auto subproblems = divide_op( std::get<0>(*data) );
       if( scheduler.is_full(subproblems.size()))
       {
         std::get<1>(*data) = seq.divide_conquer(std::get<0>(*data),
            std::forward<Divider>(divide_op), std::forward<Predicate>(predicate_op),
            std::forward<Solver>(solve_op), std::forward<Combiner>(combine_op) );
       }
       else {
         task_type merger{merger_id,task.get_task_id()};
         merger.set_data_location(task.get_data_location());
         merger.after_dependencies_ = std::move(task.after_dependencies_);
         task.after_dependencies_.clear();
         for( unsigned int i = 0; i < subproblems.size(); i++){
           bool introduced=false;
           int pos = 0;
           //Check for an empty position
           while(!introduced){
             for(int j = 0; j<1000; j++){
               if(!used[j].test_and_set()) {
                 introduced=true;
                 pos = j;
                 break;
               }
             }
           }
           //Store data in data manager
           data_manager[pos] = std::make_tuple(subproblems[i], result_type{});
           task_type t{1, scheduler.get_task_id()};
           t.set_data_location(pos);
           merger.set_task_dependency(t);
           merger.multiple_input_location.push_back(pos);
           //Launch the task
           scheduler.launch_data_task(t);
         } 
         scheduler.wait_for_dependencies(merger);
       }   
       scheduler.notify_consume();
  });
  
  

  if( predicate_op(input) ) 
     return solve_op( input);

  auto subproblems = divide_op( input );
  if( scheduler.is_full(subproblems.size()) ){ 
         return seq.divide_conquer(input,
            std::forward<Divider>(divide_op), std::forward<Predicate>(predicate_op), 
            std::forward<Solver>(solve_op), std::forward<Combiner>(combine_op) );
  }

  task_type merger{merger_id,scheduler.get_task_id()}; 
  bool introduced=false;
  while(!introduced){
    for(int j = 0; j<1000; j++){
      if(!used[j].test_and_set()) {
        introduced=true;
        merger.set_data_location(j);
        break;
      }
    }
  }

  for(unsigned int i = 0; i < subproblems.size(); i++){
    introduced=false;
    int pos = 0;
    //Check for an empty position
    while(!introduced){
      for(int j = 0; j<1000; j++){
        if(!used[j].test_and_set()) {
          introduced=true;
          pos = j;
          break;
        }
      }
    }
    //Store data in data manager
    data_manager[pos] = std::make_tuple(subproblems[i],result_type{});
    task_type t{function_id, scheduler.get_task_id()};
    t.set_data_location(pos);
    merger.set_task_dependency(t);
    merger.multiple_input_location.push_back(pos);
    //Launch the task
    scheduler.launch_data_task(t);
  } 
  scheduler.wait_for_dependencies(merger);
  
  scheduler.run_data();
  auto result = &data_manager[merger.get_data_location()];
  return std::get<1>(*result);
}
   
/*
template <typename Generator, typename ... Transformers>
void parallel_execution_task::pipeline(
    Generator && generate_op, 
    Transformers && ... transform_ops) const
{
  using namespace std;
  using result_type = decay_t<typename result_of<Generator()>::type>;
  using output_type = pair<result_type,long>;
  auto output_queue = make_queue<output_type>();

  long order=0;
  thread_pool.add_parallel_task([output_queue,&generate_op, this, &order](){
     auto item{generate_op()};
     if(item){ 
       thread_pool.tokens++;
       output_queue->push(make_pair(item, order));
       thread_pool.launch_task(0);
       thread_pool.launch_task(1);
       order++;
     } else {
        thread_pool.gen_end=true;
     }
  });

  do_pipeline(output_queue, forward<Transformers>(transform_ops)...);
}
// PRIVATE MEMBERS

template <typename Input, typename Divider, typename Solver, typename Combiner>
auto parallel_execution_task::divide_conquer(
    Input && input, 
    Divider && divide_op, 
    Solver && solve_op, 
    Combiner && combine_op,
    std::atomic<int> & num_threads) const
{
 
  auto subproblems = divide_op(std::forward<Input>(input));
  constexpr sequential_execution seq;

  num_threads++;
    return seq.divide_conquer(std::forward<Input>(input), 
        std::forward<Divider>(divide_op), std::forward<Solver>(solve_op), 
        std::forward<Combiner>(combine_op));

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

}

template <typename Input, typename Divider,typename Predicate, typename Solver, typename Combiner>
auto parallel_execution_task::divide_conquer(
    Input && input,
    Divider && divide_op,
    Predicate && predicate_op,
    Solver && solve_op,
    Combiner && combine_op,
    std::atomic<int> & num_threads) const
{
  constexpr sequential_execution seq;
  //TODO: Need to introduce a task-dependency control mechanism
  num_threads++;
//  if (num_threads.load() <=0) {
    return seq.divide_conquer(std::forward<Input>(input),
        std::forward<Divider>(divide_op),
        std::forward<Predicate>(predicate_op),
        std::forward<Solver>(solve_op),
        std::forward<Combiner>(combine_op));
*//*  }

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
  *//*
}
template <typename Queue, typename Consumer,
          requires_no_pattern<Consumer>>
void parallel_execution_task::do_pipeline(
    Queue & input_queue, 
    Consumer && consume_op) const
{
  using namespace std;
  // using input_type = typename Queue::value_type;
  //TODO: Need to reimplement ordering
  auto task_id = thread_pool.functions.size();
  thread_pool.add_sequential_task([input_queue, &consume_op, this, task_id](){
     auto item=input_queue->pop();
     consume_op(*item.first);
     thread_pool.seq_consume(task_id);
  });
  thread_pool.run();
*//*
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
  }*//*
}


template <typename Inqueue, typename Transformer, typename output_type,
            requires_no_pattern<Transformer>>
void parallel_execution_task::do_pipeline(Inqueue & input_queue, Transformer && transform_op,
      std::shared_ptr<mpmc_queue<output_type>> & output_queue) const
{
  using namespace std;
  using namespace experimental;
 
  using output_item_value_type = typename output_type::first_type::value_type;
  int task_id = thread_pool.functions.size();
  thread_pool.add_parallel_task([this, input_queue, output_queue, &transform_op, task_id](){
    auto item{input_queue->pop()}; 
    auto out = output_item_value_type{transform_op(*item.first)};
    output_queue->push(make_pair(out,item.second)) ;
    thread_pool.launch_task(task_id+1);
  });
}



template <typename Queue, typename Transformer, 
          typename ... OtherTransformers,
          requires_no_pattern<Transformer>>
void parallel_execution_task::do_pipeline(
    Queue & input_queue, 
    Transformer && transform_op,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
  using namespace experimental;

  using input_item_type = typename Queue::element_type::value_type;
  using input_item_value_type = typename input_item_type::first_type::value_type;
  using transform_result_type = 
      decay_t<typename result_of<Transformer(input_item_value_type)>::type>;
  using output_item_value_type = optional<transform_result_type>;
  using output_item_type = pair<output_item_value_type,long>;

  decltype(auto) output_queue =
    get_output_queue<output_item_type>(other_transform_ops...);

  int task_id = thread_pool.functions.size();
  thread_pool.add_sequential_task([this,output_queue,input_queue,&transform_op,task_id]() {
    auto item{input_queue->pop()};
    auto out = output_item_value_type{transform_op(*item.first)};
    output_queue->push(make_pair(out, item.second));
    thread_pool.launch_task(task_id+1);
      thread_pool.end_seq(task_id);
  });

  do_pipeline(output_queue, 
      forward<OtherTransformers>(other_transform_ops)...);
}

template <typename Queue, typename FarmTransformer,
          template <typename> class Farm,
          requires_farm<Farm<FarmTransformer>>>
void parallel_execution_task::do_pipeline(
    Queue & input_queue, 
    Farm<FarmTransformer> && farm_obj) const
{
  using namespace std;

  thread_pool.add_parallel_task([input_queue,this,&farm_obj](){
    auto item{input_queue->pop()}; 
    farm_obj(*item.first);
    thread_pool.tokens--; 
  });

  thread_pool.run();

}


template <typename Queue, typename Execution, typename Transformer,
          template <typename, typename> class Context,
          typename ... OtherTransformers,
          requires_context<Context<Execution,Transformer>>>
void parallel_execution_task::do_pipeline(Queue & input_queue, 
    Context<Execution,Transformer> && context_op,
    OtherTransformers &&... other_ops) const
{
  using namespace std;
  using namespace experimental;
*//*  using input_item_type = typename Queue::value_type;
  using input_item_value_type = typename input_item_type::first_type::value_type;

  using output_type = typename stage_return_type<input_item_value_type, Transformer>::type;
  using output_optional_type = experimental::optional<output_type>;
  using output_item_type = pair <output_optional_type, long> ;

  decltype(auto) output_queue =
    get_output_queue<output_item_type>(other_ops...);
  */
  /*auto context_task = [&]() {
    context_op.execution_policy().pipeline(input_queue, context_op.transformer(), output_queue);
    output_queue.push( make_pair(output_optional_type{}, -1) );
  };*//*
  do_pipeline(input_queue, std::forward<Transformer>(context_op.transformer()),
      forward<OtherTransformers>(other_ops)... );
  
*/ /* do_pipeline(input_queue, context_op.transformer(), output_queue);
  do_pipeline(output_queue,
      forward<OtherTransformers>(other_ops)... );*//*

}


template <typename Queue, typename FarmTransformer, 
          template <typename> class Farm,
          typename ... OtherTransformers,
          requires_farm<Farm<FarmTransformer>>>
void parallel_execution_task::do_pipeline(
    Queue & input_queue, 
    Farm<FarmTransformer> && farm_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
  using namespace experimental;

  using input_item_type = typename Queue::element_type::value_type;
  using input_item_value_type = typename input_item_type::first_type::value_type;

  using output_type = typename stage_return_type<input_item_value_type, FarmTransformer>::type;
  using output_optional_type = experimental::optional<output_type>;
  using output_item_type = pair <output_optional_type, long> ;

  decltype(auto) output_queue = 
    get_output_queue<output_item_type>(other_transform_ops...);

  
  do_pipeline(input_queue, farm_obj.transformer(), output_queue);
  do_pipeline(output_queue, 
      forward<OtherTransformers>(other_transform_ops)... );
}

template <typename Queue, typename Predicate, 
          template <typename> class Filter,
          typename ... OtherTransformers,
          requires_filter<Filter<Predicate>>>
void parallel_execution_task::do_pipeline(
    Queue & input_queue, 
    Filter<Predicate> && filter_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
  using namespace experimental;
  //TODO: to be implemented
  
  using input_item_type = typename Queue::element_type::value_type;

  auto filter_queue = make_queue<input_item_type>();
  auto task_id = thread_pool.functions.size();

  thread_pool.add_parallel_task([input_queue, filter_queue, &filter_obj, this, task_id](){
    auto item{input_queue->pop()};
      if (filter_obj(*item.first)) {
        filter_queue->push(item);
        thread_pool.launch_task(task_id+1);
      }
      else {
        thread_pool.tokens--;
      }
  });

  do_pipeline(filter_queue, forward<OtherTransformers>(other_transform_ops)...);

}

template <typename Queue, typename Combiner, typename Identity,
          template <typename C, typename I> class Reduce,
          typename ... OtherTransformers,
          requires_reduce<Reduce<Combiner,Identity>>>
void parallel_execution_task::do_pipeline(
    Queue && input_queue, 
    Reduce<Combiner,Identity> && reduce_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
  using namespace experimental;

  using output_item_value_type = optional<decay_t<Identity>>;
  using output_item_type = pair<output_item_value_type,long>;
  decltype(auto) output_queue =
    get_output_queue<output_item_type>(other_transform_ops...);

  // Review if it can be transformed into parallel task
  int task_id = thread_pool.functions.size();
  // Transform into atomic if used as a parallel task
  long int order = 0;

  thread_pool.add_sequential_task([input_queue, output_queue, &reduce_obj, this, task_id, &order](){
    auto item{input_queue->pop()};
    reduce_obj.add_item(std::forward<Identity>(*item.first));
    if(reduce_obj.reduction_needed()) {
      constexpr sequential_execution seq;
      auto red = reduce_obj.reduce_window(seq);
      output_queue->push(make_pair(red, order++));
      thread_pool.launch_task(task_id+1);
    } else{
      thread_pool.tokens--;
    }
    thread_pool.end_seq(task_id);
  });
  do_pipeline(output_queue, forward<OtherTransformers>(other_transform_ops)...);
}

template <typename Queue, typename Transformer, typename Predicate,
          template <typename T, typename P> class Iteration,
          typename ... OtherTransformers,
          requires_iteration<Iteration<Transformer,Predicate>>,
          requires_no_pattern<Transformer>>
void parallel_execution_task::do_pipeline(
    Queue & input_queue, 
    Iteration<Transformer,Predicate> && iteration_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
  using namespace experimental;
  //TODO: to be implemented
  


  using input_item_type = typename decay_t<typename Queue::element_type>::value_type;

  decltype(auto) output_queue =
    get_output_queue<input_item_type>(other_transform_ops...);

  auto task_id = thread_pool.functions.size();

  thread_pool.add_parallel_task([input_queue, output_queue, &iteration_obj, this, task_id](){
      auto item = input_queue->pop();
      auto value = iteration_obj.transform(*item.first);
      auto new_item = input_item_type{value,item.second};
      if (iteration_obj.predicate(value)) {
        output_queue->push(new_item);
        thread_pool.launch_task(task_id+1);
      }
      else {
        input_queue->push(new_item);
        thread_pool.launch_task(task_id);
      }
  });

  do_pipeline(output_queue, forward<OtherTransformers>(other_transform_ops)...);
}

template <typename Queue, typename Transformer, typename Predicate,
          template <typename T, typename P> class Iteration,
          typename ... OtherTransformers,
          requires_iteration<Iteration<Transformer,Predicate>>,
          requires_pipeline<Transformer>>
void parallel_execution_task::do_pipeline(
    Queue &,
    Iteration<Transformer,Predicate> &&,
    OtherTransformers && ...) const
{
  static_assert(!is_pipeline<Transformer>, "Not implemented");
}


template <typename Queue, typename ... Transformers,
          template <typename...> class Pipeline,
          requires_pipeline<Pipeline<Transformers...>>>
void parallel_execution_task::do_pipeline(
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
void parallel_execution_task::do_pipeline(
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
void parallel_execution_task::do_pipeline_nested(
    Queue & input_queue, 
    std::tuple<Transformers...> && transform_ops,
    std::index_sequence<I...>) const
{
  do_pipeline(input_queue,
      std::forward<Transformers>(std::get<I>(transform_ops))...);
}

template<typename T, typename... Others>
void parallel_execution_task::do_pipeline(std::shared_ptr<mpmc_queue<T>> &, std::shared_ptr<mpmc_queue<T>> &, Others &&...) const
{
}
*/
} // end namespace grppi

#endif
