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

#ifndef GRPPI_TBB_PARALLEL_EXECUTION_TBB_H
#define GRPPI_TBB_PARALLEL_EXECUTION_TBB_H

#ifdef GRPPI_TBB

#include "../common/mpmc_queue.h"
#include "../common/iterator.h"

#include <type_traits>
#include <tuple>

#include <tbb/tbb.h>

namespace grppi {

/** 
 \brief TBB parallel execution policy.

 This policy uses Intel Threading Building Blocks as implementation back end.
 */
class parallel_execution_tbb {
public:

  /** 
  \brief Default construct a TBB parallel execution policy.

  Creates a TBB parallel execution object.

  The concurrency degree is determined by the platform.

  */
  parallel_execution_tbb() noexcept :
      parallel_execution_tbb{default_concurrency_degree}
  {}

  /** 
  \brief Constructs a TBB parallel execution policy.

  Creates a TBB parallel execution object selecting the concurrency degree.

  \param concurrency_degree Number of threads used for parallel algorithms.
  \param order Whether ordered executions is enabled or disabled.
  */
  parallel_execution_tbb(int concurrency_degree, bool order = true) noexcept :
      concurrency_degree_{concurrency_degree},
      ordering_{order}
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
  \brief Sets the attributes for the queues built through make_queue<T>()
  */
  void set_queue_attributes(int size, queue_mode mode, int tokens) noexcept {
    queue_size_ = size;
    queue_mode_ = mode;
    num_tokens_ = tokens;
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
  \brien Get num of tokens.
  */
  int tokens() const noexcept { return num_tokens_; }

  /**
  \brief Applies a trasnformation to multiple sequences leaving the result in
  another sequence using available TBB parallelism.
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
  \brief Applies a trasnformation to multiple sequences leaving the result in
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

private:

  template <typename Input, typename Divider, typename Solver, typename Combiner>
  auto divide_conquer(Input && input, 
                      Divider && divide_op, 
                      Solver && solve_op, 
                      Combiner && combine_op,
                      std::atomic<int> & num_threads) const; 
private:

  constexpr static int default_concurrency_degree = 4;
  int concurrency_degree_ = default_concurrency_degree;

  bool ordering_ = true;

  constexpr static int default_queue_size = 100;
  int queue_size_ = default_queue_size;

  constexpr static int default_num_tokens_ = 100;
  int num_tokens_ = default_num_tokens_;

  queue_mode queue_mode_ = queue_mode::blocking;
};

template <typename ... InputIterators, typename OutputIterator, 
          typename Transformer>
void parallel_execution_tbb::map(
    std::tuple<InputIterators...> firsts,
    OutputIterator first_out, 
    std::size_t sequence_size, Transformer transform_op) const
{
  tbb::parallel_for(
    std::size_t{0}, sequence_size, 
    [&] (std::size_t index){
      first_out[index] = apply_iterators_indexed(transform_op, firsts, index);
    }
 );   

}

template <typename InputIterator, typename Identity, typename Combiner>
auto parallel_execution_tbb::reduce(
    InputIterator first, 
    std::size_t sequence_size,
    Identity && identity,
    Combiner && combine_op) const
{
  constexpr sequential_execution seq;
  return tbb::parallel_reduce(
      tbb::blocked_range<InputIterator>(first, std::next(first,sequence_size)),
      identity,
      [combine_op,seq](const auto & range, auto value) {
        return seq.reduce(range.begin(), range.size(), value, combine_op);
      },
      combine_op);
}

template <typename ... InputIterators, typename Identity, 
          typename Transformer, typename Combiner>
auto parallel_execution_tbb::map_reduce(
    std::tuple<InputIterators...> firsts,
    std::size_t sequence_size,
    Identity && identity,
    Transformer && transform_op, Combiner && combine_op) const
{
  constexpr sequential_execution seq;
  tbb::task_group g;

  using result_type = std::decay_t<Identity>;
  std::vector<result_type> partial_results(concurrency_degree_);

  auto process_chunk = [&](auto fins, std::size_t sz, std::size_t i) {
    partial_results[i] = seq.map_reduce(fins, sz,
        std::forward<result_type>(partial_results[i]),
        std::forward<Transformer>(transform_op), 
        std::forward<Combiner>(combine_op));
  };

  const auto chunk_size = sequence_size/concurrency_degree_;

  for(int i=0; i<concurrency_degree_-1;++i) {    
    const auto delta = chunk_size * i;
    const auto chunk_firsts = iterators_next(firsts,delta);
    const auto chunk_last = std::next(std::get<0>(chunk_firsts), chunk_size);

    g.run([&, chunk_firsts, chunk_last, i]() {
      process_chunk(chunk_firsts, chunk_size, i);
    });
  }

  const auto delta = chunk_size * (concurrency_degree_ - 1);
  const auto chunk_firsts = iterators_next(firsts,delta);
  process_chunk(chunk_firsts, sequence_size - delta, concurrency_degree_-1);

  g.wait(); 

  return seq.reduce(std::next(partial_results.begin()), 
      partial_results.size()-1, std::forward<result_type>(partial_results[0]), 
      std::forward<Combiner>(combine_op));
}

template <typename ... InputIterators, typename OutputIterator,
          typename StencilTransformer, typename Neighbourhood>
void parallel_execution_tbb::stencil(
    std::tuple<InputIterators...> firsts, OutputIterator first_out,
    std::size_t sequence_size,
    StencilTransformer && transform_op,
    Neighbourhood && neighbour_op) const
{
  constexpr sequential_execution seq{};
  const auto chunk_size = sequence_size / concurrency_degree_;
  auto process_chunk = [&](auto f, std::size_t sz, std::size_t delta) {
    seq.stencil(f, std::next(first_out,delta), sz,
      std::forward<StencilTransformer>(transform_op),
      std::forward<Neighbourhood>(neighbour_op));
  };

  tbb::task_group g;
  for (int i=0; i<concurrency_degree_-1; ++i) {
    g.run([=](){
      const auto delta = chunk_size * i;
      const auto chunk_firsts = iterators_next(firsts,delta);
      process_chunk(chunk_firsts, chunk_size, delta);
    });
  }

  const auto delta = chunk_size * (concurrency_degree_ - 1);
  const auto chunk_firsts = iterators_next(firsts,delta);
  const auto chunk_last = std::next(std::get<0>(firsts), sequence_size);
  process_chunk(chunk_firsts, 
      std::distance(std::get<0>(chunk_firsts), chunk_last), delta);

  g.wait();
}

template <typename Input, typename Divider, typename Solver, typename Combiner>
auto parallel_execution_tbb::divide_conquer(
    Input && input, 
    Divider && divide_op, 
    Solver && solve_op, 
    Combiner && combine_op) const
{
  std::atomic<int> num_threads{concurrency_degree_-1};
  return divide_conquer(std::forward<Input>(input), 
        std::forward<Divider>(divide_op), std::forward<Solver>(solve_op), 
        std::forward<Combiner>(combine_op), num_threads);
}

/**
\brief Metafunction that determines if type E is parallel_execution_tbb
\tparam Execution policy type.
*/
template <typename E>
constexpr bool is_parallel_execution_tbb() {
  return std::is_same<E, parallel_execution_tbb>::value;
}

/**
\brief Metafunction that determines if type E is supported in the current build.
\tparam Execution policy type.
*/
template <typename E>
constexpr bool is_supported();

/**
\brief Specialization stating that parallel_execution_tbb is supported.
This metafunction evaluates to false if GRPPI_TBB is enabled.
*/
template <>
constexpr bool is_supported<parallel_execution_tbb>() {
  return true;
}

// PRIVATE MEMBERS

template <typename Input, typename Divider, typename Solver, typename Combiner>
auto parallel_execution_tbb::divide_conquer(
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

  using subresult_type = std::decay_t<typename std::result_of<Solver(Input)>::type>;
  std::vector<subresult_type> partials(subproblems.size()-1);
  int division = 0;

  tbb::task_group g;
  auto i = subproblems.begin()+1;
  while (i!=subproblems.end() && num_threads.load()>0) {
    g.run([&,this,it=i++,div=division++]() {
        partials[div] = this->divide_conquer(std::forward<Input>(*it), 
            std::forward<Divider>(divide_op), std::forward<Solver>(solve_op), 
            std::forward<Combiner>(combine_op), num_threads);
    });
    num_threads--;
  }

  //Main thread works on the first subproblem.
  while (i != subproblems.end()){
    partials[division] = seq.divide_conquer(std::forward<Input>(*i++), 
        std::forward<Divider>(divide_op), std::forward<Solver>(solve_op), 
        std::forward<Combiner>(combine_op));
  }

  auto out = divide_conquer(std::forward<Input>(*subproblems.begin()),  
      std::forward<Divider>(divide_op), std::forward<Solver>(solve_op), 
      std::forward<Combiner>(combine_op), num_threads);

  g.wait();

  return seq.reduce(partials.begin(), partials.size(), 
      std::forward<subresult_type>(out), std::forward<Combiner>(combine_op));
}


} // end namespace grppi

#else // GRPPI_TBB not defined

namespace grppi {


/// Parallel execution policy.
/// Empty type if GRPPI_TBB disabled.
struct parallel_execution_tbb {};

/**
\brief Metafunction that determines if type E is parallel_execution_tbb
This metafunction evaluates to false if GRPPI_TBB is disabled.
\tparam Execution policy type.
*/
template <typename E>
constexpr bool is_parallel_execution_tbb() {
  return false;
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
constexpr bool is_supported<parallel_execution_tbb>() {
  return false;
}

} // end namespace grppi

#endif // GRPPI_TBB

#endif
