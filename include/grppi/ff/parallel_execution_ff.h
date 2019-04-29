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
#ifndef GRPPI_FF_PARALLEL_EXECUTION_FF_H
#define GRPPI_FF_PARALLEL_EXECUTION_FF_H

#ifdef GRPPI_FF

#include "grppi/ff/detail/pipeline_impl.h"

#include "../common/iterator.h"
#include "../common/execution_traits.h"

#include <type_traits>
#include <tuple>
#include <thread>

#include <ff/parallel_for.hpp>
#include <ff/dc.hpp>

namespace grppi {

/**
\brief FastFlow (FF) parallel execution policy.

This policy uses FF as implementation back-end.
*/
class parallel_execution_ff {

public:
  /** 
  \brief Default construct a FF parallel execution policy.

  Creates an FF parallel execution object.

  The concurrency degree is determined by the platform.
  */
  parallel_execution_ff() noexcept :
      parallel_execution_ff{
        static_cast<int>(std::thread::hardware_concurrency())}
  {}

  /** 
  \brief Constructs an FF parallel execution policy.

  Creates an FF parallel execution object selecting the concurrency degree
  and ordering.
  \param concurrency_degree Number of threads used for parallel algorithms.
  \param order Whether ordered executions is enabled or disabled.
  */
  parallel_execution_ff(int concurrency_degree, bool order = true) noexcept :
    concurrency_degree_{concurrency_degree}, 
    ordering_{order}
  {
  }

  /**
  \brief Set number of grppi threads.
  */
  void set_concurrency_degree(int degree) noexcept { 
    concurrency_degree_ = degree;
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
  \brief Applies a trasnformation to multiple sequences leaving the result in
  another sequence using available FF parallelism
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
    \param last Iterator to one past the end of the sequence.
    \param identity Identity value for the reduction.
    \param combine_op Combination callable object.
    \pre Iterators in the range `[first,last)` are valid.
    \return The reduction result.
   */
  template <typename InputIterator, typename Identity, typename Combiner>
  auto reduce(InputIterator first,
      std::size_t sequence_size,
      Identity && identity,
      Combiner && combine_op) const;

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
      Transformer && transform_op,
      Combiner && combine_op) const;

  /**
    \brief Applies a transformation to multiple sequences leaving the result in
    another sequence.
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
            typename StencilTransformer, typename Neighbourhood>
  void stencil(std::tuple<InputIterators...> firsts,
      OutputIterator first_out,
      std::size_t sequence_size,
      StencilTransformer && transform_op,
      Neighbourhood && neighbour_op) const;

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


  /**
  \brief Invoke \ref md_pipeline comming from another context
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
                mpmc_queue<OutputType> & output_queue) const
  {
    ::std::atomic<long> order {0};
    pipeline(
      [&](){
        auto item = input_queue.pop();
        if(!item.first) input_queue.push(item);
        return item.first;
      },
      std::forward<Transformer>(transform_op),
      [&](auto & item ){
        output_queue.push(make_pair(typename OutputType::first_type{item}, order.load()));
        order++;
      }
    );
    output_queue.push(make_pair(typename OutputType::first_type{}, order.load()));
  }

  /**
    \brief Invoke \ref md_divide-conquer.
    \tparam Input Type used for the input problem.
    \tparam Divider Callable type for the divider operation.
    \tparam Solver Callable type for the solver operation.
    \tparam Combiner Callable type for the combiner operation.
    \param input Input problem to be solved.
    \param divider_op Divider operation.
    \param solver_op Solver operation.
    \param combine_op Combiner operation.
   */
  template <typename Input, typename Divider,typename Predicate, 
            typename Solver, typename Combiner>
  auto divide_conquer(Input & input,
      Divider && divide_op,
      Predicate && condition_op,
      Solver && solve_op,
      Combiner && combine_op) const;

private:

  int concurrency_degree_ = 
    static_cast<int>(std::thread::hardware_concurrency());
  bool ordering_ = true;
};

/**
\brief Metafunction that determines if type E is parallel_execution_ff
\tparam Execution policy type.
*/
template <typename E>
constexpr bool is_parallel_execution_ff() {
  return std::is_same<E, parallel_execution_ff>::value;
}

/**
\brief Determines if an execution policy is supported in the current compilation.
\note Specialization for parallel_execution_ff when GRPPI_FF is enabled.
*/
template <>
constexpr bool is_supported<parallel_execution_ff>() { return true; }

/**
\brief Determines if an execution policy supports the map pattern.
\note Specialization for parallel_execution_ff when GRPPI_FF is enabled.
*/
template <>
constexpr bool supports_map<parallel_execution_ff>() { return true; }

/**
\brief Determines if an execution policy supports the reduce pattern.
\note Specialization for parallel_execution_ff when GRPPI_FF is enabled.
*/
template <>
constexpr bool supports_reduce<parallel_execution_ff>() { return true; }

/**
\brief Determines if an execution policy supports the map-reduce pattern.
\note Specialization for parallel_execution_ff when GRPPI_FF is enabled.
*/
template <>
constexpr bool supports_map_reduce<parallel_execution_ff>() { return true; }

/**
\brief Determines if an execution policy supports the stencil pattern.
\note Specialization for parallel_execution_ff when GRPPI_FF is enabled.
*/
template <>
constexpr bool supports_stencil<parallel_execution_ff>() { return true; }

/*
\brief Determines if an execution policy supports the divide_conquer pattern.
\note Specialization for parallel_execution_ff when GRPPI_FF is enabled.
*/
template <>
constexpr bool supports_divide_conquer<parallel_execution_ff>() { return true; }

/**
\brief Determines if an execution policy supports the pipeline pattern.
\note Specialization for parallel_execution_ff when GRPPI_FF is enabled.
*/
template <>
constexpr bool supports_pipeline<parallel_execution_ff>() { return true; }


template <typename ... InputIterators, typename OutputIterator, 
          typename Transformer>
void parallel_execution_ff::map(
    std::tuple<InputIterators...> firsts,
    OutputIterator first_out, 
    std::size_t sequence_size, Transformer transform_op) const
{
  ff::ParallelFor pf{concurrency_degree_, true};
  pf.parallel_for(0, sequence_size,
    [=](const long delta) {
      *std::next(first_out, delta) = apply_iterators_indexed(transform_op, firsts, delta);
    }, 
    concurrency_degree_);
}

template <typename InputIterator, typename Identity, typename Combiner>
auto parallel_execution_ff::reduce(InputIterator first,
    std::size_t sequence_size,
    Identity && identity,
    Combiner && combine_op) const 
{
  ff::ParallelForReduce<Identity> pfr{concurrency_degree_, true};
  Identity result{identity};

  pfr.parallel_reduce(result, identity, 0, sequence_size,
      [combine_op,first](long delta, auto & value) {
        value = combine_op(value, *std::next(first,delta));
      }, 
      [&result, combine_op](auto a, auto b) { result = combine_op(a,b); }, 
      concurrency_degree_);

  return result;
}

template <typename ... InputIterators, typename Identity,
          typename Transformer, typename Combiner>
auto parallel_execution_ff::map_reduce(std::tuple<InputIterators...> firsts,
    std::size_t sequence_size,
    Identity && identity,
    Transformer && transform_op,
    Combiner && combine_op) const 
{
  std::vector<Identity> partial_outs(sequence_size);
  map(firsts, partial_outs.begin(), sequence_size, 
      std::forward<Transformer>(transform_op));

  return reduce(partial_outs.begin(), sequence_size, 
      std::forward<Identity>(identity),
      std::forward<Combiner>(combine_op));
}

template <typename ... InputIterators, typename OutputIterator,
          typename StencilTransformer, typename Neighbourhood>
void parallel_execution_ff::stencil(std::tuple<InputIterators...> firsts,
    OutputIterator first_out,
    std::size_t sequence_size,
    StencilTransformer && transform_op,
    Neighbourhood && neighbour_op) const 
{
  ff::ParallelFor pf(concurrency_degree_, true);
  pf.parallel_for(0, sequence_size,
    [&](long delta) {
      const auto first_it = std::get<0>(firsts);
      auto next_chunks = iterators_next(firsts, delta);
      *std::next(first_out,delta) = transform_op(std::next(first_it,delta),
          apply_increment(neighbour_op, next_chunks) );
    }, 
    concurrency_degree_);
}

template <typename Generator, typename ... Transformers>
void parallel_execution_ff::pipeline(
    Generator && generate_op,
    Transformers && ... transform_ops) const 
{
  detail_ff::pipeline_impl pipe{
      concurrency_degree_, 
      ordering_,
      std::forward<Generator>(generate_op),
      std::forward<Transformers>(transform_ops)...};

  pipe.setFixedSize(false);
  pipe.run_and_wait_end();
}

template <typename Input, typename Divider,typename Predicate, 
          typename Solver, typename Combiner>
auto parallel_execution_ff::divide_conquer(Input & input,
    Divider && divide_op,
    Predicate && condition_op,
    Solver && solve_op,
    Combiner && combine_op) const 
{
  using output_type = typename std::result_of<Solver(Input)>::type;
  
  // divide
  auto divide_fn = [&](const Input &in, std::vector<Input> &subin) {
    subin = divide_op(in);
  };
  // combine
  auto combine_fn = [&] (std::vector<output_type>& in, output_type& out) {
	  using index_t = typename std::vector<output_type>::size_type;
	  out = in[0];
	  for(index_t i = 1; i < in.size(); ++i)
		  out = combine_op(out, in[i]);
  };
  // sequential solver (base-case)
  auto seq_fn = [&] (const Input & in , output_type & out) {
    out = solve_op(in);
  };
  // condition
  auto cond_fn = [&] (const Input &in) {
    return condition_op(in);
  };
  output_type out_var{};

  using dac_t = ff::ff_DC<Input,output_type>;
  auto ncores = static_cast<int>(std::thread::hardware_concurrency());
  int max_nworkers = std::max(concurrency_degree_, ncores);
  dac_t dac(divide_fn, combine_fn, seq_fn, cond_fn, //kernel functions
			input, out_var, //input/output variables
			concurrency_degree_, //parallelism degree
			dac_t::DEFAULT_OUTSTANDING_TASKS, max_nworkers //ff-specific params
			);

  // run
  dac.run_and_wait_end();

  return out_var;
}

} // end namespace grppi

#else // GRPPI_FF undefined

namespace grppi {


/// Parallel execution policy.
/// Empty type if  GRPPI_FF disabled.
struct parallel_execution_ff {};

/**
\brief Metafunction that determines if type E is parallel_execution_ff
This metafunction evaluates to false if GRPPI_FF is disabled.
\tparam Execution policy type.
*/
template <typename E>
constexpr bool is_parallel_execution_ff() {
  return false;
}

}

#endif // GRPPI_FF

#endif
