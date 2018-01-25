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

#ifndef GRPPI_OMP_PARALLEL_EXECUTION_OMP_H
#define GRPPI_OMP_PARALLEL_EXECUTION_OMP_H

#ifdef GRPPI_OMP

#include "../common/mpmc_queue.h"
#include "../common/iterator.h"
#include "../common/execution_traits.h"
#include "../seq/sequential_execution.h"

#include <type_traits>
#include <tuple>
#include <experimental/optional>

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
  \brief Returns the reference of a communication queue for elements of type T 
  if the queue has been created in an outer pattern.
  Returns the reference of the queue received as argument.
  \tparam T Element type for the queue.
  \tparam Transformers List of the next transformers.
  \param queue Reference of a queue of type T
  */
  template <typename T, typename ... Transformers>
  mpmc_queue<T>& get_output_queue(mpmc_queue<T> & queue, Transformers ...) const {
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
  mpmc_queue<T> get_output_queue(Transformers ... ) const{
    return std::move(make_queue<T>());
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

  /**
  \brief Applies a trasnformation to multiple sequences leaving the result in
  another sequence using available OpenMP parallelism
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
  template <typename Input, typename Divider, typename Predicate, typename Solver, typename Combiner>
  auto divide_conquer(Input && input,
		      Divider && divide_op,
		      Predicate && condition_op,
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
                Transformers && ... transform_op) const;

private:

  template <typename Input, typename Divider, typename Solver, typename Combiner>
  auto divide_conquer(Input && input, 
                      Divider && divide_op, 
                      Solver && solve_op, 
                      Combiner && combine_op,
                      std::atomic<int> & num_threads) const;

  template <typename Input, typename Divider, typename Predicate, typename Solver, typename Combiner>
  auto divide_conquer(Input && input,
		      Divider && divide_op,
		      Predicate && condition_op,
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
  void do_pipeline(mpmc_queue<T> & in_q, mpmc_queue<T> & same_queue, Others &&... ops) const
  { }
   
  template <typename T>
  void do_pipeline(mpmc_queue<T> & in_q) const {}


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
  void do_pipeline(Queue & input_queue, 
                   Farm<FarmTransformer> && farm_obj) const;

  template <typename Queue, typename FarmTransformer, 
            template <typename> class Farm,
            typename ... OtherTransformers,
            requires_farm<Farm<FarmTransformer>> =0>
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
            requires_farm<Farm<FarmTransformer>> =0>
  void do_pipeline(Queue & input_queue, 
       Farm<FarmTransformer> && farm_obj,
       OtherTransformers && ... other_transform_ops) const;

  template <typename Queue, typename Predicate,
            template <typename> class Filter,
            requires_filter<Filter<Predicate>> = 0>
  void do_pipeline(Queue & input_queue, 
                   Filter<Predicate> & filter_obj) const
  {
    do_pipeline(input_queue, std::move(filter_obj));
  }

  template <typename Queue, typename Predicate,
            template <typename> class Filter,
            requires_filter<Filter<Predicate>> = 0>
  void do_pipeline(Queue & input_queue, 
                   Filter<Predicate> && filter_obj) const;

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
       Filter<Predicate> && filter_obj,
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
\brief Determines if an execution policy is supported in the current compilation.
\note Specialization for parallel_execution_omp when GRPPI_OMP is enabled.
*/
template <>
constexpr bool is_supported<parallel_execution_omp>() { return true; }

/**
\brief Determines if an execution policy supports the map pattern.
\note Specialization for parallel_execution_omp when GRPPI_OMP is enabled.
*/
template <>
constexpr bool supports_map<parallel_execution_omp>() { return true; }

/**
\brief Determines if an execution policy supports the reduce pattern.
\note Specialization for parallel_execution_omp when GRPPI_OMP is enabled.
*/
template <>
constexpr bool supports_reduce<parallel_execution_omp>() { return true; }

/**
\brief Determines if an execution policy supports the map-reduce pattern.
\note Specialization for parallel_execution_omp when GRPPI_OMP is enabled.
*/
template <>
constexpr bool supports_map_reduce<parallel_execution_omp>() { return true; }

/**
\brief Determines if an execution policy supports the stencil pattern.
\note Specialization for parallel_execution_omp when GRPPI_OMP is enabled.
*/
template <>
constexpr bool supports_stencil<parallel_execution_omp>() { return true; }

/**
\brief Determines if an execution policy supports the divide/conquer pattern.
\note Specialization for parallel_execution_omp when GRPPI_OMP is enabled.
*/
template <>
constexpr bool supports_divide_conquer<parallel_execution_omp>() { return true; }

/**
\brief Determines if an execution policy supports the pipeline pattern.
\note Specialization for parallel_execution_omp when GRPPI_OMP is enabled.
*/
template <>
constexpr bool supports_pipeline<parallel_execution_omp>() { return true; }

template <typename ... InputIterators, typename OutputIterator, 
          typename Transformer>
void parallel_execution_omp::map(
    std::tuple<InputIterators...> firsts,
    OutputIterator first_out, 
    std::size_t sequence_size, Transformer transform_op) const
{
  #pragma omp parallel for
  for (std::size_t i=0; i<sequence_size; ++i) {
    first_out[i] = apply_iterators_indexed(transform_op, firsts, i);
  }
}

template <typename InputIterator, typename Identity, typename Combiner>
auto parallel_execution_omp::reduce(
    InputIterator first, std::size_t sequence_size,
    Identity && identity,
    Combiner && combine_op) const
{
  constexpr sequential_execution seq;

  using result_type = std::decay_t<Identity>;
  std::vector<result_type> partial_results(concurrency_degree_);
  auto process_chunk = [&](InputIterator f, std::size_t sz, std::size_t id) {
    partial_results[id] = seq.reduce(f, sz, std::forward<Identity>(identity), 
        std::forward<Combiner>(combine_op));
  };

  const auto chunk_size = sequence_size/concurrency_degree_;

  #pragma omp parallel
  {
    #pragma omp single nowait
    {
      for (int i=0 ;i<concurrency_degree_-1; ++i) {
        const auto delta = chunk_size * i;
        const auto chunk_first = std::next(first,delta);

        #pragma omp task firstprivate (chunk_first, chunk_size, i)
        {
          process_chunk(chunk_first, chunk_size, i);
        }
      }
    
      //Main thread
      const auto delta = chunk_size * (concurrency_degree_ - 1);
      const auto chunk_first= std::next(first,delta);
      const auto chunk_sz = sequence_size - delta;
      process_chunk(chunk_first, chunk_sz, concurrency_degree_-1);
      #pragma omp taskwait
    }
  }

  return seq.reduce(std::next(partial_results.begin()), 
      partial_results.size()-1,
      partial_results[0], std::forward<Combiner>(combine_op));
}

template <typename ... InputIterators, typename Identity, 
          typename Transformer, typename Combiner>
auto parallel_execution_omp::map_reduce(
    std::tuple<InputIterators...> firsts,
    std::size_t sequence_size,
    Identity && identity,
    Transformer && transform_op, Combiner && combine_op) const
{
  constexpr sequential_execution seq;

  using result_type = std::decay_t<Identity>;
  std::vector<result_type> partial_results(concurrency_degree_);

  auto process_chunk = [&](auto f, std::size_t sz, std::size_t i) {
    partial_results[i] = seq.map_reduce(
        f, sz, partial_results[i],
        std::forward<Transformer>(transform_op), 
        std::forward<Combiner>(combine_op));
  };

  const auto chunk_size = sequence_size / concurrency_degree_;

  #pragma omp parallel
  {
    #pragma omp single nowait
    {

      for (int i=0;i<concurrency_degree_-1;++i) {    
        #pragma omp task firstprivate(i)
        {
          const auto delta = chunk_size * i;
          const auto chunk_firsts = iterators_next(firsts,delta);
          const auto chunk_last = std::next(std::get<0>(chunk_firsts), chunk_size);
          process_chunk(chunk_firsts, chunk_size, i);
        }
      }

      const auto delta = chunk_size * (concurrency_degree_ - 1);
      auto chunk_firsts = iterators_next(firsts,delta);
      auto chunk_last = std::next(std::get<0>(firsts), sequence_size);
      process_chunk(chunk_firsts, 
          std::distance(std::get<0>(chunk_firsts), chunk_last), 
          concurrency_degree_ - 1);
      #pragma omp taskwait
    }
  }

  return seq.reduce(std::next(partial_results.begin()), 
      partial_results.size()-1,
      partial_results[0], std::forward<Combiner>(combine_op));
}

template <typename ... InputIterators, typename OutputIterator,
          typename StencilTransformer, typename Neighbourhood>
void parallel_execution_omp::stencil(
    std::tuple<InputIterators...> firsts, OutputIterator first_out,
    std::size_t sequence_size,
    StencilTransformer && transform_op,
    Neighbourhood && neighbour_op) const
{
  constexpr sequential_execution seq;
  const auto chunk_size = sequence_size / concurrency_degree_;
  auto process_chunk = [&](auto f, std::size_t sz, std::size_t delta) {
    seq.stencil(f, std::next(first_out,delta), sz,
      std::forward<StencilTransformer>(transform_op),
      std::forward<Neighbourhood>(neighbour_op));
  };

  #pragma omp parallel 
  {
    #pragma omp single nowait
    {
      for (int i=0; i<concurrency_degree_-1; ++i) {
        #pragma omp task firstprivate(i)
        {
          const auto delta = chunk_size * i;
          const auto chunk_firsts = iterators_next(firsts,delta);
          process_chunk(chunk_firsts, chunk_size, delta);
        }
      }

      const auto delta = chunk_size * (concurrency_degree_ - 1);
      const auto chunk_firsts = iterators_next(firsts,delta);
      const auto chunk_last = std::next(std::get<0>(firsts), sequence_size);
      process_chunk(chunk_firsts, 
          std::distance(std::get<0>(chunk_firsts), chunk_last), delta);

      #pragma omp taskwait
    }
  }
}

template <typename Input, typename Divider, typename Solver, typename Combiner>
auto parallel_execution_omp::divide_conquer(
    Input && input, 
    Divider && divide_op, 
    Solver && solve_op, 
    Combiner && combine_op) const
{
  std::atomic<int> num_threads{concurrency_degree_-1};
  
  return divide_conquer(std::forward<Input>(input), 
      std::forward<Divider>(divide_op), std::forward<Solver>(solve_op), 
      std::forward<Combiner>(combine_op),
      num_threads);
}

template <typename Input, typename Divider,typename Predicate,
  typename Solver, typename Combiner>
auto parallel_execution_omp::divide_conquer(Input && input,
					    Divider && divide_op,
					    Predicate && condition_op,
					    Solver && solve_op,
					    Combiner && combine_op) const
 {
   std::atomic<int> num_threads{concurrency_degree_-1};

   return divide_conquer(std::forward<Input>(input),
			 std::forward<Divider>(divide_op), std::forward<Predicate>(condition_op),
			 std::forward<Solver>(solve_op),
			 std::forward<Combiner>(combine_op),
			 num_threads);
 }

template <typename Generator, typename ... Transformers>
void parallel_execution_omp::pipeline(
    Generator && generate_op, 
    Transformers && ... transform_ops) const
{
  using namespace std;

  using result_type = decay_t<typename result_of<Generator()>::type>;
  auto output_queue = make_queue<pair<result_type,long>>(); 

  #pragma omp parallel
  {
    #pragma omp single nowait
    {
      #pragma omp task shared(generate_op,output_queue)
      {
        long order = 0;
        for (;;) {
          auto item = generate_op();
          output_queue.push(make_pair(item,order++)) ;
          if (!item) break;
        }
      }
      do_pipeline(output_queue,
          forward<Transformers>(transform_ops)...);
      #pragma omp taskwait
    }
  }
}

// PRIVATE MEMBERS
template <typename Input, typename Divider, typename Solver, typename Combiner>
auto parallel_execution_omp::divide_conquer(
    Input && input, 
    Divider && divide_op, 
    Solver && solve_op, 
    Combiner && combine_op,
    std::atomic<int> & num_threads) const
{
  constexpr sequential_execution seq;
  if (num_threads.load()<=0) {
    return seq.divide_conquer(std::forward<Input>(input), 
        std::forward<Divider>(divide_op), std::forward<Solver>(solve_op), 
        std::forward<Combiner>(combine_op));
  }

  auto subproblems = divide_op(std::forward<Input>(input));
  if (subproblems.size()<=1) { return solve_op(std::forward<Input>(input)); }

  using subresult_type = 
      std::decay_t<typename std::result_of<Solver(Input)>::type>;
  std::vector<subresult_type> partials(subproblems.size()-1);

  auto process_subproblems = [&,this](auto it, std::size_t div) {
    partials[div] = this->divide_conquer(std::forward<Input>(*it), 
        std::forward<Divider>(divide_op), std::forward<Solver>(solve_op), 
        std::forward<Combiner>(combine_op), num_threads);
  };

  int division = 0;
  subresult_type subresult;

  #pragma omp parallel
  {
    #pragma omp single nowait
    { 
      auto i = subproblems.begin() + 1;
      while (i!=subproblems.end() && num_threads.load()>0) {
        #pragma omp task firstprivate(i,division) \
                shared(partials,divide_op,solve_op,combine_op,num_threads)
        {
           process_subproblems(i, division);
        }
        num_threads --;
        i++;
        division++;
      }

      while (i!=subproblems.end()) { 
        partials[division] = seq.divide_conquer(std::forward<Input>(*i++), 
          std::forward<Divider>(divide_op), std::forward<Solver>(solve_op), 
          std::forward<Combiner>(combine_op));
      }

      //Main thread works on the first subproblem.
      if (num_threads.load()>0) {
        subresult = divide_conquer(std::forward<Input>(*subproblems.begin()), 
            std::forward<Divider>(divide_op), std::forward<Solver>(solve_op), 
            std::forward<Combiner>(combine_op), num_threads);
      }
      else {
        subresult = seq.divide_conquer(std::forward<Input>(*subproblems.begin()), 
            std::forward<Divider>(divide_op), std::forward<Solver>(solve_op), 
            std::forward<Combiner>(combine_op));
      }
      #pragma omp taskwait
    }
  }
  return seq.reduce(partials.begin(), partials.size(), 
      std::forward<subresult_type>(subresult), combine_op);
}

template <typename Input, typename Divider,typename Predicate, typename Solver, typename Combiner>
auto parallel_execution_omp::divide_conquer(Input && input,
					    Divider && divide_op,
					    Predicate && condition_op,
					    Solver && solve_op,
					    Combiner && combine_op,
					    std::atomic<int> & num_threads) const
 {
   constexpr sequential_execution seq;
   if (num_threads.load()<=0) {
     return seq.divide_conquer(std::forward<Input>(input),
			       std::forward<Divider>(divide_op),std::forward<Predicate>(condition_op),
			       std::forward<Solver>(solve_op),
			       std::forward<Combiner>(combine_op));
   }

   if (condition_op(input)) { return solve_op(std::forward<Input>(input)); }
   auto subproblems = divide_op(std::forward<Input>(input));

     using subresult_type =
       std::decay_t<typename std::result_of<Solver(Input)>::type>;
     std::vector<subresult_type> partials(subproblems.size()-1);

     auto process_subproblems = [&,this](auto it, std::size_t div) {
       partials[div] = this->divide_conquer(std::forward<Input>(*it),
					    std::forward<Divider>(divide_op), std::forward<Predicate>(condition_op),
					    std::forward<Solver>(solve_op),
					    std::forward<Combiner>(combine_op), num_threads);
     };

     int division = 0;
     subresult_type subresult;

       #pragma omp parallel
     {
           #pragma omp single nowait
       {
	 auto i = subproblems.begin() + 1;
	 while (i!=subproblems.end() && num_threads.load()>0) {
#pragma omp task firstprivate(i,division) \
  shared(partials,divide_op,solve_op,combine_op,num_threads)
	   {
	     process_subproblems(i, division);
	   }
	   num_threads --;
	   i++;
	   division++;
	 }

	 while (i!=subproblems.end()) {
	   partials[division] = seq.divide_conquer(std::forward<Input>(*i++),
						   std::forward<Divider>(divide_op), std::forward<Predicate>(condition_op),
						   std::forward<Solver>(solve_op),
						   std::forward<Combiner>(combine_op));
	 }

	 //Main thread works on the first subproblem.
	 if (num_threads.load()>0) {
	   subresult = divide_conquer(std::forward<Input>(*subproblems.begin()),
				      std::forward<Divider>(divide_op),std::forward<Predicate>(condition_op),
				      std::forward<Solver>(solve_op),
				      std::forward<Combiner>(combine_op), num_threads);
	 }
	 else {
	   subresult = seq.divide_conquer(std::forward<Input>(*subproblems.begin()),
					  std::forward<Divider>(divide_op), std::forward<Predicate>(condition_op),
					  std::forward<Solver>(solve_op),
					  std::forward<Combiner>(combine_op));
	 }
	       #pragma omp taskwait
       }
     }
     return seq.reduce(partials.begin(), partials.size(),
		       std::forward<subresult_type>(subresult), combine_op);
 }

template <typename Queue, typename Consumer,
          requires_no_pattern<Consumer> =0>
void parallel_execution_omp::do_pipeline(Queue & input_queue, Consumer && consume_op) const
{
  using namespace std;
  using input_type = typename Queue::value_type;

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
  auto item = input_queue.pop( );
  while (item.first) {
    if (current == item.second) {
      consume_op(*item.first);
      current ++;
    } 
    else {
      elements.push_back(item);
    }
    for (auto it=elements.begin(); it!=elements.end(); it++) {
      if (it->second == current) {
        consume_op(*it->first);
        elements.erase(it);
        current++;
        break;
      }
    }
    item = input_queue.pop( );
  }
  while(elements.size()>0){
    for(auto it = elements.begin(); it != elements.end(); it++){
      if(it->second == current) {
        consume_op(*it->first);
        elements.erase(it);
        current++;
        break;
      }
    }
  }
}


template <typename Inqueue, typename Transformer, typename output_type,
            requires_no_pattern<Transformer> = 0>
void parallel_execution_omp::do_pipeline(Inqueue & input_queue, Transformer && transform_op,
      mpmc_queue<output_type> & output_queue) const
{
  using namespace std;
  using namespace experimental;

  using input_item_type = typename Inqueue::value_type;
  using input_item_value_type = typename input_item_type::first_type::value_type;

  using output_item_value_type = typename output_type::first_type::value_type;
  for (;;) {
    auto item{input_queue.pop()}; 
    if(!item.first) break;
    auto out = output_item_value_type{transform_op(*item.first)};
    output_queue.push(make_pair(out,item.second)) ;
  }
}

template <typename Queue, typename Transformer, typename ... OtherTransformers,
          requires_no_pattern<Transformer> =0>
void parallel_execution_omp::do_pipeline(
    Queue & input_queue, 
    Transformer && transform_op,
    OtherTransformers && ... other_ops) const
{
  using namespace std;
  using input_type = typename Queue::value_type;
  using input_value_type = typename input_type::first_type::value_type;
  using result_type = typename result_of<Transformer(input_value_type)>::type;
  using output_value_type = experimental::optional<result_type>;
  using output_type = pair<output_value_type,long>;

  decltype(auto) output_queue =
    get_output_queue<output_type>(other_ops...);

  #pragma omp task shared(transform_op, input_queue, output_queue)
  {
    for (;;) {
      auto item = input_queue.pop(); 
      if (!item.first) break;
      auto out = output_value_type{transform_op(*item.first)};
      output_queue.push(make_pair(out, item.second));
    }
    output_queue.push(make_pair(output_value_type{}, -1));
  }

  do_pipeline(output_queue, 
      forward<OtherTransformers>(other_ops)...);
}

template <typename Queue, typename FarmTransformer,
          template <typename> class Farm,
          requires_farm<Farm<FarmTransformer>> =0>
void parallel_execution_omp::do_pipeline(
    Queue & input_queue, 
    Farm<FarmTransformer> && farm_obj) const
{
  using namespace std;
  using namespace experimental;
  using input_type = typename Queue::value_type;
  using input_value_type = typename input_type::first_type::value_type;
 
  for (int i=0; i<farm_obj.cardinality(); ++i) {
    #pragma omp task shared(farm_obj,input_queue)
    {
      auto item = input_queue.pop();
      while (item.first) {
        farm_obj(*item.first);
        item = input_queue.pop();
      }
      input_queue.push(item);
    }              
  }
  #pragma omp taskwait
}

template <typename Queue, typename FarmTransformer, 
          template <typename> class Farm,
          typename ... OtherTransformers,
          requires_farm<Farm<FarmTransformer>> =0>
void parallel_execution_omp::do_pipeline(
    Queue & input_queue, 
    Farm<FarmTransformer> && farm_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
  using namespace experimental;
  using input_type = typename Queue::value_type;
  using input_value_type = typename input_type::first_type::value_type;

  using result_type = typename stage_return_type<input_value_type, FarmTransformer>::type;
  using output_optional_type = optional<result_type>;
  using output_type = pair<output_optional_type,long>;
 
  decltype(auto) output_queue =
    get_output_queue<output_type>(other_transform_ops...);

//  auto output_queue = make_queue<output_type>();

  atomic<int> done_threads{0};
  int ntask = farm_obj.cardinality();
  for (int i=0; i<farm_obj.cardinality(); ++i) {
    #pragma omp task shared(done_threads,output_queue,farm_obj,input_queue,ntask)
    {
      do_pipeline(input_queue, farm_obj.transformer(), output_queue);
      done_threads++;
      if (done_threads == ntask){
        output_queue.push(make_pair(output_optional_type{}, -1));
      }else{
        input_queue.push(input_type{});
      }
    }              
  }
  do_pipeline(output_queue, forward<OtherTransformers>(other_transform_ops)...);
  #pragma omp taskwait
}


template <typename Queue, typename Predicate,
          template <typename> class Filter,
          requires_filter<Filter<Predicate>> = 0>
void parallel_execution_omp::do_pipeline(
    Queue & input_queue, 
    Filter<Predicate> && filter_obj) const
{
}

template <typename Queue, typename Predicate, 
          template <typename> class Filter,
          typename ... OtherTransformers,
          requires_filter<Filter<Predicate>> =0>
void parallel_execution_omp::do_pipeline(
    Queue & input_queue, 
    Filter<Predicate> && filter_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
  using input_type = typename Queue::value_type;
  using input_value_type = typename input_type::first_type;
  auto filter_queue = make_queue<input_type>();

  if (is_ordered()) {
    auto filter_task = [&]() {
      {
        auto item{input_queue.pop()};
        while (item.first) {
          if(filter_obj(*item.first)) {
            filter_queue.push(item);
          }
          else {
            filter_queue.push(make_pair(input_value_type{} ,item.second));
          }
          item = input_queue.pop();
        }
        filter_queue.push (make_pair(input_value_type{}, -1));
      }
    };

    decltype(auto) output_queue =
      get_output_queue<input_type>(other_transform_ops...);


    auto reorder_task = [&]() {
      vector<input_type> elements;
      int current = 0;
      long order = 0;
      auto item = filter_queue.pop();
      for (;;) {
        if (!item.first && item.second == -1) break;
        if (item.second == current) {
          if (item.first) {
            output_queue.push(make_pair(item.first, order++));
          }
          current++;
        }
        else {
          elements.push_back(item);
        }
        for (auto it=elements.begin(); it<elements.end(); it++) {
          if ((*it).second==current) {
            if((*it).first){
              output_queue.push(make_pair((*it).first,order++));
            }
            elements.erase(it);
            current++;
            break;
          }
        }
        item = filter_queue.pop();
      }

      while (elements.size()>0) {
        for (auto it=elements.begin(); it<elements.end(); it++) {
          if ((*it).second == current) {
            if((*it).first) {
              output_queue.push(make_pair((*it).first,order++));
            }
            elements.erase(it);
            current++;
            break;
          }
        }
      }

      output_queue.push(item);
    };

    
    #pragma omp task shared(filter_queue,filter_obj,input_queue)
    {
      filter_task();
    }
    
    #pragma omp task shared (output_queue,filter_queue)
    {
      reorder_task();
    }

    do_pipeline(output_queue, 
        forward<OtherTransformers>(other_transform_ops)...);

    #pragma omp taskwait
  }
  else {
    auto filter_task = [&]() {
      auto item = input_queue.pop( ) ;
      while (item.first) {
        if (filter_obj(*item.first)) {
          filter_queue.push(item);
        }
        item = input_queue.pop();
      }
      filter_queue.push(make_pair(input_value_type{}, -1));
    };

    #pragma omp task shared(filter_queue,filter_obj,input_queue)
    {
      filter_task();
    }
    do_pipeline(filter_queue, 
      std::forward<OtherTransformers>(other_transform_ops)...);
    #pragma omp taskwait
  }
}


template <typename Queue, typename Combiner, typename Identity,
          template <typename C, typename I> class Reduce,
          typename ... OtherTransformers,
          requires_reduce<Reduce<Combiner,Identity>> = 0>
void parallel_execution_omp::do_pipeline(
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
  
  decltype(auto) output_queue =
    get_output_queue<output_item_type>(other_transform_ops...);

  auto reduce_task = [&,this]() {
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

  #pragma omp task shared(reduce_obj,input_queue, output_queue)
  {
    reduce_task();
  }
  do_pipeline(output_queue, 
      std::forward<OtherTransformers>(other_transform_ops)...);
  #pragma omp taskwait
}

template <typename Queue, typename Transformer, typename Predicate,
          template <typename T, typename P> class Iteration,
          typename ... OtherTransformers,
          requires_iteration<Iteration<Transformer,Predicate>> =0,
          requires_no_pattern<Transformer> =0>
void parallel_execution_omp::do_pipeline(
    Queue & input_queue, 
    Iteration<Transformer,Predicate> && iteration_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
  using namespace experimental;

  using input_item_type = typename decay_t<Queue>::value_type;
  using input_item_value_type = typename input_item_type::first_type::value_type;
  decltype(auto) output_queue =
    get_output_queue<input_item_type>(other_transform_ops...);


  auto iteration_task = [&]() {
    for (;;) {
      auto item = input_queue.pop();
      if (!item.first) break;
      std::cerr << "Processing: <" << *item.first << " , " << item.second << ">\n";
      auto value = iteration_obj.transform(*item.first);
      auto new_item = input_item_type{value,item.second};
      if (iteration_obj.predicate(value)) {
        std::cerr << "Sending to output"
            << *new_item.first << " , " << new_item.second << ">\n";
        output_queue.push(new_item);
      }
      else {
        std::cerr << "Sending to input"
            << *new_item.first << " , " << new_item.second << ">\n";
        input_queue.push(new_item);
      }
    }
    while (!input_queue.is_empty()) {
      auto item = input_queue.pop();
      std::cerr << "Processing: <" << *item.first << " , " << item.second << ">\n";
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

  #pragma omp task shared(iteration_obj,input_queue,output_queue)
  {
    iteration_task();
  }
  do_pipeline(output_queue, 
      std::forward<OtherTransformers>(other_transform_ops)...);
  #pragma omp taskwait

}

template <typename Queue, typename Transformer, typename Predicate,
          template <typename T, typename P> class Iteration,
          typename ... OtherTransformers,
          requires_iteration<Iteration<Transformer,Predicate>> =0,
          requires_pipeline<Transformer> =0>
void parallel_execution_omp::do_pipeline(
    Queue & input_queue, 
    Iteration<Transformer,Predicate> && iteration_obj,
    OtherTransformers && ... other_transform_ops) const
{
  static_assert(!is_pipeline<Transformer>, "Not implemented");
}

template <typename Queue, typename ... Transformers,
          template <typename...> class Pipeline,
          typename ... OtherTransformers,
          requires_pipeline<Pipeline<Transformers...>> = 0>
void parallel_execution_omp::do_pipeline(
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
void parallel_execution_omp::do_pipeline_nested(
    Queue & input_queue, 
    std::tuple<Transformers...> && transform_ops,
    std::index_sequence<I...>) const
{
  do_pipeline(input_queue,
      std::forward<Transformers>(std::get<I>(transform_ops))...);
}


} // end namespace grppi

#else // GRPPI_OMP undefined

namespace grppi {


/// Parallel execution policy.
/// Empty type if  GRPPI_OMP disabled.
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

}

#endif // GRPPI_OMP

#endif
