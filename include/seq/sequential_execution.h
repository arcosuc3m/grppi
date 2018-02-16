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

#ifndef GRPPI_SEQ_SEQUENTIAL_EXECUTION_H
#define GRPPI_SEQ_SEQUENTIAL_EXECUTION_H

#include "../common/mpmc_queue.h"
#include "../common/iterator.h"
#include "../common/callable_traits.h"
#include "../common/execution_traits.h"
#include "../common/patterns.h"
#include "../common/pack_traits.h"

#include <type_traits>
#include <tuple>
#include <iterator>
//#include <experimental/optional>

namespace grppi {

/**
\brief Sequential execution policy.
*/
class sequential_execution {

public:

  /// \brief Default constructor.
  constexpr sequential_execution() noexcept = default;

  /**
  \brief Set number of grppi threads.
  \note Setting concurrency degree is ignored for sequential execution.
  */
  constexpr void set_concurrency_degree(int n) const noexcept {}

  /**
  \brief Get number of grppi trheads.
  \note Getting concurrency degree is always 1 for sequential execution.
  */
  constexpr int concurrency_degree() const noexcept { return 1; }

  /**
  \brief Enable ordering.
  \note Enabling ordering of sequential execution is always ignored.
  */
  constexpr void enable_ordering() const noexcept {}

  /**
  \brief Disable ordering.
  \note Disabling ordering of sequential execution is always ignored.
  */
  constexpr void disable_ordering() const noexcept {}

  /**
  \brief Is execution ordered.
  \note Sequential execution is always ordered.
  */
  constexpr bool is_ordered() const noexcept { return true; }

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
            typename Transformer>
  constexpr void map(std::tuple<InputIterators...> firsts,
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
  constexpr auto reduce(InputIterator first, std::size_t sequence_size,
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
  constexpr auto map_reduce(std::tuple<InputIterators...> firsts, 
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
  constexpr void stencil(std::tuple<InputIterators...> firsts, OutputIterator first_out,
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
  template <typename Input, typename Divider,typename Predicate, typename Solver, typename Combiner>
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
    using namespace std;
    using optional_output_type = typename OutputType::first_type;
    for(;;){
      auto item = input_queue.pop();
      if(!item.first) break;
      do_pipeline(*item.first, std::forward<Transformer>(transform_op),
        [&](auto output_item) {
          output_queue.push( make_pair(optional_output_type{output_item}, item.second) );
        }
      );
    }
    output_queue.push( make_pair(optional_output_type{}, -1) );
  }


private:

  template <typename Item, typename Consumer>
  concept_no_pattern<Consumer,void>
  do_pipeline(Item && item, Consumer && consume_op) const;

  template <typename Item, typename Transformer, typename ... OtherTransformers>
  concept_no_pattern<Transformer,void>
  do_pipeline(Item && item, Transformer && transform_op,
          OtherTransformers && ... other_ops) const;

  template <typename Item, typename FarmTransformer,
            template <typename> class Farm>
  concept_farm<Farm<FarmTransformer>,void>
  do_pipeline(Item && item, Farm<FarmTransformer> & farm_obj) const
  {
    do_pipeline(std::forward<Item>(item), std::move(farm_obj));
  }

  template <typename Item, typename FarmTransformer,
            template <typename> class Farm>
  concept_farm<Farm<FarmTransformer>,void>
  do_pipeline(Item && item, Farm<FarmTransformer> && farm_obj) const;

  template <typename Item, typename Execution, typename Transformer,
            template <typename, typename> class Context,
            typename ... OtherTransformers>
  concept_context<Context<Execution,Transformer>,void>
  do_pipeline(Item && item, Context<Execution,Transformer> && context_op,
       OtherTransformers &&... other_ops) const
  {
     do_pipeline(item, std::forward<Transformer>(context_op.transformer()), 
       std::forward<OtherTransformers>(other_ops)...);
  }

  template <typename Item, typename Execution, typename Transformer,
            template <typename, typename> class Context,
            typename ... OtherTransformers>
  concept_context<Context<Execution,Transformer>,void>
  do_pipeline(Item && item, Context<Execution,Transformer> & context_op,
       OtherTransformers &&... other_ops) const
  {
    do_pipeline(item, std::move(context_op),
      std::forward<OtherTransformers>(other_ops)...);
  }


  template <typename Item, typename FarmTransformer,
            template <typename> class Farm,
            typename... OtherTransformers>
  concept_farm<Farm<FarmTransformer>,void>
  do_pipeline(Item && item, Farm<FarmTransformer> & farm_obj,
                   OtherTransformers && ... other_transform_ops) const
  {
    do_pipeline(std::forward<Item>(item), std::move(farm_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename Item, typename FarmTransformer,
            template <typename> class Farm,
            typename... OtherTransformers>
  concept_farm<Farm<FarmTransformer>,void>
  do_pipeline(Item && item, Farm<FarmTransformer> && farm_obj,
                   OtherTransformers && ... other_transform_ops) const;

  template <typename Item, typename Predicate,
            template <typename> class Filter,
            typename ... OtherTransformers>
  concept_filter<Filter<Predicate>,void>
  do_pipeline(Item && item, Filter<Predicate> & filter_obj,
                   OtherTransformers && ... other_transform_ops) const
  {
    do_pipeline(std::forward<Item>(item), std::move(filter_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename Item, typename Predicate,
            template <typename> class Filter,
            typename ... OtherTransformers>
  concept_filter<Filter<Predicate>,void>
  do_pipeline(Item && item, Filter<Predicate> && filter_obj,
                   OtherTransformers && ... other_transform_ops) const;

  template <typename Item, typename Combiner, typename Identity,
            template <typename C, typename I> class Reduce,
            typename ... OtherTransformers>
  concept_reducer<Reduce<Combiner,Identity>,void>
  do_pipeline(Item && item, Reduce<Combiner,Identity> & reduce_obj,
                   OtherTransformers && ... other_transform_ops) const
  {
    do_pipeline(std::forward<Item>(item), std::move(reduce_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  };
    

  template <typename Item, typename Combiner, typename Identity,
            template <typename C, typename I> class Reduce,
            typename ... OtherTransformers>
  concept_reducer<Reduce<Combiner,Identity>,void>
  do_pipeline(Item && item, Reduce<Combiner,Identity> && reduce_obj,
                   OtherTransformers && ... other_transform_ops) const;

  template <typename Item, typename Transformer, typename Predicate,
            template <typename T, typename P> class Iteration,
            typename ... OtherTransformers>
  concept_iteration<Iteration<Transformer,Predicate>,void>
  do_pipeline(Item && item, 
          Iteration<Transformer,Predicate> & iteration_obj, 
          OtherTransformers && ... other_transform_ops) const
  {
    do_pipeline(std::forward<Item>(item), std::move(iteration_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename Item, typename Transformer, typename Predicate,
            template <typename T, typename P> class Iteration,
            typename ...OtherTransformers>
  concept_iteration_plain<Iteration<Transformer,Predicate>, Transformer,void>
  do_pipeline(Item && item, 
          Iteration<Transformer,Predicate> && iteration_obj, 
          OtherTransformers && ... other_transform_ops) const;

  template <typename Item, typename Transformer, typename Predicate,
            template <typename T, typename P> class Iteration,
            typename ...OtherTransformers>
  concept_iteration_pipeline<Iteration<Transformer,Predicate>,Transformer,void>
  do_pipeline(Item && item, 
          Iteration<Transformer,Predicate> && iteration_obj, 
          OtherTransformers && ... other_transform_ops) const;

  template <typename Item, typename ... Transformers,
            template <typename...> class Pipeline,
            typename ... OtherTransformers,
            requires_pipeline<Pipeline<Transformers...>> = 0>
  concept_pipeline<Pipeline<Transformers...>,void>
  do_pipeline(Item && item, Pipeline<Transformers...> & pipeline_obj,
                   OtherTransformers && ... other_transform_ops) const
  {
    do_pipeline(std::forward<Item>(item), std::move(pipeline_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename Item, typename ... Transformers,
            template <typename...> class Pipeline,
            typename ... OtherTransformers,
            requires_pipeline<Pipeline<Transformers...>> = 0>
  concept_pipeline<Pipeline<Transformers...>,void>
  do_pipeline(Item && item, Pipeline<Transformers...> && pipeline_obj,
                   OtherTransformers && ... other_transform_ops) const;

  template <typename Item, typename ... Transformers, std::size_t ... I>
  void do_pipeline_nested(Item && item, 
          std::tuple<Transformers...> && transform_ops,
          std::index_sequence<I...>) const;

};

/// Determine if a type is a sequential execution policy.
template <typename E>
constexpr bool is_sequential_execution() {
  return std::is_same<E, sequential_execution>::value;
}

/**
\brief Determines if an execution policy is supported in the current compilation.
\note Specialization for sequential_execution.
*/
template <>
constexpr bool is_supported<sequential_execution>() { return true; }

/**
\brief Determines if an execution policy supports the map pattern.
\note Specialization for sequential_execution.
*/
template <>
constexpr bool supports_map<sequential_execution>() { return true; }

/**
\brief Determines if an execution policy supports the reduce pattern.
\note Specialization for sequential_execution.
*/
template <>
constexpr bool supports_reduce<sequential_execution>() { return true; }

/**
\brief Determines if an execution policy supports the map-reduce pattern.
\note Specialization for sequential_execution.
*/
template <>
constexpr bool supports_map_reduce<sequential_execution>() { return true; }

/**
\brief Determines if an execution policy supports the stencil pattern.
\note Specialization for sequential_execution.
*/
template <>
constexpr bool supports_stencil<sequential_execution>() { return true; }

/**
\brief Determines if an execution policy supports the divide/conquer pattern.
\note Specialization for sequential_execution.
*/
template <>
constexpr bool supports_divide_conquer<sequential_execution>() { return true; }

/**
\brief Determines if an execution policy supports the pipeline pattern.
\note Specialization for sequential_execution.
*/
template <>
constexpr bool supports_pipeline<sequential_execution>() { return true; }

template <typename ... InputIterators, typename OutputIterator,
          typename Transformer>
constexpr void sequential_execution::map(
    std::tuple<InputIterators...> firsts,
    OutputIterator first_out, 
    std::size_t sequence_size, 
    Transformer && transform_op) const
{
  const auto last = std::next(std::get<0>(firsts), sequence_size);
  while (std::get<0>(firsts) != last) {
    *first_out++ = apply_deref_increment(
        std::forward<Transformer>(transform_op), firsts);
  }
}

template <typename InputIterator, typename Identity, typename Combiner>
constexpr auto sequential_execution::reduce(
    InputIterator first, 
    std::size_t sequence_size,
    Identity && identity,
    Combiner && combine_op) const
{
  const auto last = std::next(first, sequence_size);
  auto result{identity};
  while (first != last) {
    result = combine_op(result, *first++);
  }
  return result;
}

template <typename ... InputIterators, typename Identity, 
          typename Transformer, typename Combiner>
constexpr auto sequential_execution::map_reduce(
    std::tuple<InputIterators...> firsts,
    std::size_t sequence_size, 
    Identity && identity,
    Transformer && transform_op, Combiner && combine_op) const
{
  const auto last = std::next(std::get<0>(firsts), sequence_size);
  auto result{identity};
  while (std::get<0>(firsts) != last) {
    result = combine_op(result, apply_deref_increment(
        std::forward<Transformer>(transform_op), firsts));
  }
  return result;
}

template <typename ... InputIterators, typename OutputIterator,
          typename StencilTransformer, typename Neighbourhood>
constexpr void sequential_execution::stencil(
    std::tuple<InputIterators...> firsts, OutputIterator first_out,
    std::size_t sequence_size,
    StencilTransformer && transform_op,
    Neighbourhood && neighbour_op) const
{
  const auto last = std::next(std::get<0>(firsts), sequence_size);
  while (std::get<0>(firsts) != last) {
    const auto f = std::get<0>(firsts);
    *first_out++ = transform_op(f, 
        apply_increment(std::forward<Neighbourhood>(neighbour_op), firsts));
  }
}


template <typename Input, typename Divider, typename Predicate, typename Solver, typename Combiner>
auto sequential_execution::divide_conquer(
    Input && input,
    Divider && divide_op,
    Predicate && predicate_op,
    Solver && solve_op,
    Combiner && combine_op) const
{

  if (predicate_op(input)) { return solve_op(std::forward<Input>(input)); }
  auto subproblems = divide_op(std::forward<Input>(input));

  using subproblem_type =
      std::decay_t<typename std::result_of<Solver(Input)>::type>;
  std::vector<subproblem_type> solutions;
  for (auto && sp : subproblems) {
    solutions.push_back(divide_conquer(sp,
        std::forward<Divider>(divide_op), std::forward<Predicate>(predicate_op),std::forward<Solver>(solve_op),
        std::forward<Combiner>(combine_op)));
  }
  return reduce(std::next(solutions.begin()), solutions.size()-1, solutions[0],
      std::forward<Combiner>(combine_op));
}


template <typename Input, typename Divider, typename Solver, typename Combiner>
auto sequential_execution::divide_conquer(
    Input && input, 
    Divider && divide_op, 
    Solver && solve_op, 
    Combiner && combine_op) const
{

  auto subproblems = divide_op(std::forward<Input>(input));
  if (subproblems.size()<=1) { return solve_op(std::forward<Input>(input)); }

  using subproblem_type = 
      std::decay_t<typename std::result_of<Solver(Input)>::type>;
  std::vector<subproblem_type> solutions;
  for (auto && sp : subproblems) {
    solutions.push_back(divide_conquer(sp, 
        std::forward<Divider>(divide_op), std::forward<Solver>(solve_op), 
        std::forward<Combiner>(combine_op)));
  }
  return reduce(std::next(solutions.begin()), solutions.size()-1, solutions[0],
      std::forward<Combiner>(combine_op));
}

template <typename Generator, typename ... Transformers>
void sequential_execution::pipeline(
    Generator && generate_op,
    Transformers && ... transform_ops) const
{
  static_assert(is_generator<Generator>,
    "First pipeline stage must be a generator");

  for (;;) {
    auto x = generate_op();
    if (!x) break;
    do_pipeline(*x, std::forward<Transformers>(transform_ops)...);
  }
}

template <typename Item, typename Consumer>
concept_no_pattern<Consumer,void>
sequential_execution::do_pipeline(
    Item && item,
    Consumer && consume_op) const
{
  consume_op(std::forward<Item>(item));
}

template <typename Item, typename Transformer, typename ... OtherTransformers>
concept_no_pattern<Transformer,void>
sequential_execution::do_pipeline(
    Item && item,
    Transformer && transform_op,
    OtherTransformers && ... other_ops) const
{
  static_assert(!is_consumer<Transformer,Item>,
    "Itermediate pipeline stage cannot be a consumer");

  do_pipeline(transform_op(std::forward<Item>(item)), 
      std::forward<OtherTransformers>(other_ops)...);
}

template <typename Item, typename FarmTransformer,
          template <typename> class Farm>
concept_farm<Farm<FarmTransformer>,void>
sequential_execution::do_pipeline(
    Item && item, 
    Farm<FarmTransformer> && farm_obj) const
{
  farm_obj(std::forward<Item>(item));
}

template <typename Item, typename FarmTransformer,
          template <typename> class Farm,
          typename... OtherTransformers>
concept_farm<Farm<FarmTransformer>,void>
sequential_execution::do_pipeline(
    Item && item, 
    Farm<FarmTransformer> && farm_obj,
    OtherTransformers && ... other_transform_ops) const
{
  static_assert(!is_consumer<Farm<FarmTransformer>,Item>,
    "Itermediate pipeline stage cannot be a consumer");
  do_pipeline(farm_obj(std::forward<Item>(item)), 
      std::forward<OtherTransformers>(other_transform_ops)...);
}

template <typename Item, typename Predicate,
          template <typename> class Filter,
          typename ... OtherTransformers>
concept_filter<Filter<Predicate>,void>
sequential_execution::do_pipeline(
    Item && item, 
    Filter<Predicate> && filter_obj,
    OtherTransformers && ... other_transform_ops) const
{
  if (filter_obj(std::forward<Item>(item))) {
    do_pipeline(std::forward<Item>(item),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }
}

template <typename Item, typename Combiner, typename Identity,
          template <typename C, typename I> class Reduce,
          typename ... OtherTransformers>
concept_reducer<Reduce<Combiner,Identity>,void>
sequential_execution::do_pipeline(
    Item && item, 
    Reduce<Combiner,Identity> && reduce_obj,
    OtherTransformers && ... other_transform_ops) const
{
  reduce_obj.add_item(std::forward<Identity>(item));
  if (reduce_obj.reduction_needed()) {
    auto red = reduce_obj.reduce_window(*this);
    do_pipeline(red,
        std::forward<OtherTransformers...>(other_transform_ops)...);
  }
}

template <typename Item, typename Transformer, typename Predicate,
          template <typename T, typename P> class Iteration,
          typename ... OtherTransformers>
concept_iteration_plain<Iteration<Transformer,Predicate>,Transformer,void>
sequential_execution::do_pipeline(
    Item && item, 
    Iteration<Transformer,Predicate> && iteration_obj, 
    OtherTransformers && ... other_transform_ops) const
{
  auto new_item = iteration_obj.transform(std::forward<Item>(item));
  while (!iteration_obj.predicate(new_item)) {
    new_item = iteration_obj.transform(new_item);
  }
  do_pipeline(new_item,
      std::forward<OtherTransformers...>(other_transform_ops)...);
}

template <typename Item, typename Transformer, typename Predicate,
          template <typename T, typename P> class Iteration,
          typename ... OtherTransformers>
concept_iteration_pipeline<Iteration<Transformer,Predicate>,Transformer,void>
sequential_execution::do_pipeline(
    Item && item, 
    Iteration<Transformer,Predicate> && iteration_obj, 
    OtherTransformers && ... other_transform_ops) const
{
  static_assert(!is_pipeline<Transformer>, "Not implemented");
}

template <typename Item, typename ... Transformers,
          template <typename...> class Pipeline,
          typename ... OtherTransformers,
          requires_pipeline<Pipeline<Transformers...>> = 0>
concept_pipeline<Pipeline<Transformers...>,void>
sequential_execution::do_pipeline(
    Item && item, 
    Pipeline<Transformers...> && pipeline_obj,
    OtherTransformers && ... other_transform_ops) const
{
  do_pipeline_nested(
      std::forward<Item>(item),
      std::tuple_cat(pipeline_obj.transformers(), 
          std::forward_as_tuple(other_transform_ops...)),
      std::make_index_sequence<sizeof...(Transformers)+sizeof...(OtherTransformers)>());
}

template <typename Item, typename ... Transformers, std::size_t ... I>
void sequential_execution::do_pipeline_nested(
    Item && item, 
    std::tuple<Transformers...> && transform_ops,
    std::index_sequence<I...>) const
{
  do_pipeline(
      std::forward<Item>(item),
      std::forward<Transformers>(std::get<I>(transform_ops))...);
}

} // end namespace grppi

#endif
