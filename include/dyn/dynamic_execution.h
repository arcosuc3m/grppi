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
#ifndef GRPPI_DYN_DYNAMIC_EXECUTION_H
#define GRPPI_DYN_DYNAMIC_EXECUTION_H

#include "../seq/sequential_execution.h"
#include "../native/parallel_execution_native.h"

#include "../task/parallel_execution_task.h"
#include "../task/fifo_scheduler.h"
#include "../task/simple_task.h"

#include "../tbb/parallel_execution_tbb.h"
#include "../omp/parallel_execution_omp.h"
#include "../ff/parallel_execution_ff.h"
#include "../common/configuration.h"

#include <memory>

namespace grppi{

class dynamic_execution {
public:

  dynamic_execution() noexcept : execution_{} 
  {
    configuration<> config;
    switch (config.dynamic_backend()) {
      case execution_backend::seq:
        execution_ = make_execution<sequential_execution>();
        break;
      case execution_backend::native:
        execution_ = make_execution<parallel_execution_native>();
        break;
      case execution_backend::omp:
        execution_ = make_execution<parallel_execution_omp>();
        break;
      case execution_backend::tbb:
        execution_ = make_execution<parallel_execution_tbb>();
        break;
      case execution_backend::ff:
        execution_ = make_execution<parallel_execution_ff>();
        break;
      case execution_backend::task:
        execution_ = make_execution<parallel_execution_task<fifo_scheduler<simple_task>>>();
        break;
    }
  }

  template <typename E>
  dynamic_execution(const E & e) : execution_{std::make_unique<execution<E>>(e)} {}

  bool has_execution() const { return execution_.get() != nullptr; }

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
            typename Transformer>
  void map(std::tuple<InputIterators...> firsts,
          OutputIterator first_out, std::size_t sequence_size, 
          Transformer && transform_op) const;
  
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
  \return The reduction result
  */
  template <typename InputIterator, typename Identity, typename Combiner>
  auto reduce(InputIterator first, std::size_t sequence_size,
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

private:
  
  class execution_base {
  public:
    virtual ~execution_base() {};
  };

  template <typename E>
  class execution : public execution_base {
  public:
    execution(const E & e) : ex_{e} {}
    virtual ~execution() = default;
    E ex_;
  };

  template <typename E>
  std::unique_ptr<execution<E>> make_execution() {
    return std::make_unique<execution<E>>(E{});
  }

private:
  /// Pointer to dynamically allocated execution policy.
  std::unique_ptr<execution_base> execution_;

};

/**
\brief Determines if an execution policy is supported in the current compilation.
\note Specialization for dynamic_execution.
*/
template <>
constexpr bool is_supported<dynamic_execution>() { return true; }

/**
\brief Determines if an execution policy supports the map pattern.
\note Specialization for dynamic_execution.
*/
template <>
constexpr bool supports_map<dynamic_execution>() { return true; }

/**
\brief Determines if an execution policy supports the reduce pattern.
\note Specialization for dynamic_execution.
*/
template <>
constexpr bool supports_reduce<dynamic_execution>() { return true; }

/**
\brief Determines if an execution policy supports the map-reduce pattern.
\note Specialization for dynamic_execution.
*/
template <>
constexpr bool supports_map_reduce<dynamic_execution>() { return true; }

/**
\brief Determines if an execution policy supports the stencil pattern.
\note Specialization for dynamic_execution.
*/
template <>
constexpr bool supports_stencil<dynamic_execution>() { return true; }

/**
\brief Determines if an execution policy supports the divide/conquer pattern.
\note Specialization for dynamic_execution.
*/
template <>
constexpr bool supports_divide_conquer<dynamic_execution>() { return true; }

/**
\brief Determines if an execution policy supports the pipeline pattern.
\note Specialization for dynamic_execution.
*/
template <>
constexpr bool supports_pipeline<dynamic_execution>() { return true; }

#define GRPPI_TRY_PATTERN(E,PATTERN,...)\
{\
  if (supports_##PATTERN<E>()) {\
    auto * ex = dynamic_cast<execution<E>*>(execution_.get());\
    if (ex) {\
      return ex->ex_.PATTERN(__VA_ARGS__);\
    }\
  }\
}

#define GRPPI_PATTERN_NOT_IMPLEMENTED(PATTERN,...)\
throw std::runtime_error{"Pattern " #PATTERN " not implemented"};

#ifdef GRPPI_OMP
#define GRPPI_TRY_PATTERN_OMP(PATTERN,...) \
GRPPI_TRY_PATTERN(parallel_execution_omp,PATTERN,__VA_ARGS__)
#else
#define GRPPI_TRY_PATTERN_OMP(PATTERN,...)
#endif

#ifdef GRPPI_TBB
#define GRPPI_TRY_PATTERN_TBB(PATTERN,...) \
GRPPI_TRY_PATTERN(parallel_execution_tbb,PATTERN,__VA_ARGS__)
#else
#define GRPPI_TRY_PATTERN_TBB(PATTERN,...)
#endif

#ifdef GRPPI_FF
#define GRPPI_TRY_PATTERN_FF(PATTERN,...) \
GRPPI_TRY_PATTERN(parallel_execution_ff,PATTERN,__VA_ARGS__)
#else
#define GRPPI_TRY_PATTERN_FF(PATTERN,...)
#endif

#define GRPPI_TRY_PATTERN_ALL(...) \
GRPPI_TRY_PATTERN(sequential_execution, __VA_ARGS__) \
GRPPI_TRY_PATTERN(parallel_execution_native, __VA_ARGS__) \
GRPPI_TRY_PATTERN(parallel_execution_task<fifo_scheduler<simple_task>>, __VA_ARGS__) \
GRPPI_TRY_PATTERN_OMP(__VA_ARGS__) \
GRPPI_TRY_PATTERN_TBB(__VA_ARGS__) \
GRPPI_TRY_PATTERN_FF(__VA_ARGS__) \
GRPPI_PATTERN_NOT_IMPLEMENTED(__VA_ARGS__)\

#define GRPPI_TRY_PATTERN_ALL_NOFF(...) \
GRPPI_TRY_PATTERN(sequential_execution, __VA_ARGS__) \
GRPPI_TRY_PATTERN(parallel_execution_native, __VA_ARGS__) \
GRPPI_TRY_PATTERN(parallel_execution_task<fifo_scheduler<simple_task>>, __VA_ARGS__) \
GRPPI_TRY_PATTERN_OMP(__VA_ARGS__) \
GRPPI_TRY_PATTERN_TBB(__VA_ARGS__) \
GRPPI_PATTERN_NOT_IMPLEMENTED(__VA_ARGS__)\

template <typename ... InputIterators, typename OutputIterator, 
          typename Transformer>
void dynamic_execution::map(
    std::tuple<InputIterators...> firsts,
    OutputIterator first_out, 
    std::size_t sequence_size, 
    Transformer && transform_op) const 
{
  GRPPI_TRY_PATTERN_ALL(map, firsts, first_out, sequence_size, 
      std::forward<Transformer>(transform_op));
}

template <typename InputIterator, typename Identity, typename Combiner>
auto dynamic_execution::reduce(InputIterator first, std::size_t sequence_size,
          Identity && identity,
          Combiner && combine_op) const
{
  GRPPI_TRY_PATTERN_ALL(reduce, first, sequence_size, 
      std::forward<Identity>(identity), std::forward<Combiner>(combine_op));
}

template <typename ... InputIterators, typename Identity, 
          typename Transformer, typename Combiner>
auto dynamic_execution::map_reduce(
    std::tuple<InputIterators...> firsts, 
    std::size_t sequence_size,
    Identity && identity,
    Transformer && transform_op, 
    Combiner && combine_op) const
{
  GRPPI_TRY_PATTERN_ALL(map_reduce, firsts, sequence_size, 
      std::forward<Identity>(identity), 
      std::forward<Transformer>(transform_op),
      std::forward<Combiner>(combine_op));
}

template <typename ... InputIterators, typename OutputIterator,
          typename StencilTransformer, typename Neighbourhood>
void dynamic_execution::stencil(
    std::tuple<InputIterators...> firsts, 
    OutputIterator first_out,
    std::size_t sequence_size,
    StencilTransformer && transform_op,
    Neighbourhood && neighbour_op) const
{
  GRPPI_TRY_PATTERN_ALL(stencil, firsts, first_out, sequence_size,
      std::forward<StencilTransformer>(transform_op),
      std::forward<Neighbourhood>(neighbour_op));
}

template <typename Input, typename Divider, typename Solver, typename Combiner>
auto dynamic_execution::divide_conquer(
    Input && input, 
    Divider && divide_op, 
    Solver && solve_op, 
    Combiner && combine_op) const
{
  GRPPI_TRY_PATTERN_ALL_NOFF(divide_conquer, std::forward<Input>(input),
      std::forward<Divider>(divide_op),
      std::forward<Solver>(solve_op),
      std::forward<Combiner>(combine_op));
}


template <typename Input, typename Divider,typename Predicate, typename Solver, typename Combiner>
auto dynamic_execution::divide_conquer(
    Input && input,
    Divider && divide_op,
    Predicate && predicate_op,
    Solver && solve_op,
    Combiner && combine_op) const
{
  GRPPI_TRY_PATTERN_ALL(divide_conquer, std::forward<Input>(input),
      std::forward<Divider>(divide_op),
      std::forward<Predicate>(predicate_op),
      std::forward<Solver>(solve_op),
      std::forward<Combiner>(combine_op));
}

template <typename Generator, typename ... Transformers>
void dynamic_execution::pipeline(
    Generator && generate_op, 
    Transformers && ... transform_ops) const
{
  GRPPI_TRY_PATTERN_ALL(pipeline, std::forward<Generator>(generate_op),
      std::forward<Transformers>(transform_ops)...);
}

#undef GRPPI_TRY_PATTERN
#undef GRPPI_TRY_PATTERN_OMP
#undef GRPPI_TRY_PATTERN_TBB
#undef GRPPI_TRY_PATTERN_ALL
#undef GRPPI_TRY_PATTERN_ALL_NOFF

} // end namespace grppi

#endif
