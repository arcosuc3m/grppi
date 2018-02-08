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

#ifndef GRPPI_FF_PARALLEL_EXECUTION_FF_H
#define GRPPI_FF_PARALLEL_EXECUTION_FF_H

#ifdef GRPPI_FF

#include "../common/iterator.h"
#include "../common/execution_traits.h"

#include <type_traits>
#include <tuple>
#include <thread>
#include <experimental/optional>

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
  \note The concurrency_degree is a hint and the implementation might select
        a lower value.
  */
  parallel_execution_ff(int concurrency_degree, bool order = true) noexcept :
      concurrency_degree_{std::min(concurrency_degree, 
          static_cast<int>(std::thread::hardware_concurrency()))},
      ordering_{order}
  {
  }

  /**
  \brief Set number of grppi threads.
  \note The concurrency_degree is a hint and the implementation might select
        a lower value.
  */
  void set_concurrency_degree(int degree) noexcept { 
    concurrency_degree_ = std::min(degree,
        static_cast<int>(std::thread::hardware_concurrency())); 
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

/**
\brief Determines if an execution policy supports the divide_conquer pattern.
\note Specialization for parallel_execution_ff when GRPPI_FF is enabled.
*/
template <>
constexpr bool supports_divide_conquer<parallel_execution_ff>() { return true; }


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
      out = combine_op(in[0], in[1]);
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
  // divide, combine, seq, condition, input, res, wrks
  ff::ff_DC<Input,output_type> dac(divide_fn, combine_fn, seq_fn, cond_fn, input, out_var, concurrency_degree_);
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
