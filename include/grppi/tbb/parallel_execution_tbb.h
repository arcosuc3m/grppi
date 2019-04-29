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
#ifndef GRPPI_TBB_PARALLEL_EXECUTION_TBB_H
#define GRPPI_TBB_PARALLEL_EXECUTION_TBB_H

#ifdef GRPPI_TBB

#include "../common/optional.h"
#include "../common/mpmc_queue.h"
#include "../common/iterator.h"
#include "../common/patterns.h"
#include "../common/farm_pattern.h"
#include "../common/execution_traits.h"

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
  parallel_execution_tbb() noexcept 
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
  void set_concurrency_degree(int degree) noexcept { 
    concurrency_degree_ = degree; 
    num_tokens_ = token_factor_ * concurrency_degree_;
  }

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
  \brief Applies a transformation to multiple sequences leaving the result in
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
    //sequential_execution seq{};
    //seq.pipeline(input_queue, std::forward<Transformer>(transform_op), output_queue);
  }


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

  template <typename Input, typename Divider, typename Predicate, typename Solver, typename Combiner>
  auto divide_conquer(Input && input,
                      Divider && divide_op,
                      Predicate && predicate_op,
                      Solver && solve_op,
                      Combiner && combine_op,
                      std::atomic<int> & num_threads) const;

  template <typename Input, typename Transformer, 
            requires_no_pattern<Transformer> = 0>
  auto make_filter(Transformer && transform_op) const;

  template <typename Input, typename Transformer, typename ... OtherTransformers,
            requires_no_pattern<Transformer> = 0>
  auto make_filter(Transformer && transform_op,
                   OtherTransformers && ... other_transform_ops) const;

  template <typename Input, typename FarmTransformer,
            template <typename> class Farm,
            requires_farm<Farm<FarmTransformer>> = 0>
  auto make_filter(Farm<FarmTransformer> & farm_obj) const
  {
    return this->template make_filter<Input>(std::move(farm_obj));
  }

  template <typename Input, typename FarmTransformer,
            template <typename> class Farm,
            requires_farm<Farm<FarmTransformer>> = 0>
  auto make_filter(Farm<FarmTransformer> && farm_obj) const;

  template <typename Input, typename Execution, typename Transformer,
            template <typename, typename> class Context,
            typename ... OtherTransformers,
            requires_context<Context<Execution,Transformer>> = 0>
  auto make_filter(Context<Execution,Transformer> && context_op,
       OtherTransformers &&... other_ops) const
  {
     return this->template make_filter<Input>(std::forward<Transformer>(context_op.transformer()),
       std::forward<OtherTransformers>(other_ops)...);
  }

  template <typename Input, typename Execution, typename Transformer,
            template <typename, typename> class Context,
            typename ... OtherTransformers,
            requires_context<Context<Execution,Transformer>> = 0>
  auto make_filter(Context<Execution,Transformer> & context_op,
       OtherTransformers &&... other_ops) const
  {
    return this->template make_filter<Input>(std::move(context_op),
      std::forward<OtherTransformers>(other_ops)...);
  }


  template <typename Input, typename FarmTransformer, 
            template <typename> class Farm,
            typename ... OtherTransformers,
            requires_farm<Farm<FarmTransformer>> = 0>
  auto make_filter(Farm<FarmTransformer> & filter_obj,
                   OtherTransformers && ... other_transform_ops) const
  {
    return this->template make_filter<Input>(std::move(filter_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename Input, typename FarmTransformer, 
            template <typename> class Farm,
            typename ... OtherTransformers,
            requires_farm<Farm<FarmTransformer>> = 0>
  auto make_filter(Farm<FarmTransformer> && filter_obj,
                   OtherTransformers && ... other_transform_ops) const;

  template <typename Input, typename Predicate,
            template <typename> class Filter,
            requires_filter<Filter<Predicate>> = 0>
  auto make_filter(Filter<Predicate> & filter_obj) const
  {
    return this->template make_filter<Input>(std::move(filter_obj));
  }

  template <typename Input, typename Predicate,
            template <typename> class Filter,
            requires_filter<Filter<Predicate>> = 0>
  auto make_filter(Filter<Predicate> && filter_obj) const;

  template <typename Input, typename Predicate, 
            template <typename> class Filter,
            typename ... OtherTransformers,
            requires_filter<Filter<Predicate>> = 0>
  auto make_filter(Filter<Predicate> & filter_obj,
                   OtherTransformers && ... other_transform_ops) const
  {
    return this->template make_filter<Input>(std::move(filter_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename Input, typename Predicate, 
            template <typename> class Filter,
            typename ... OtherTransformers,
            requires_filter<Filter<Predicate>> = 0>
  auto make_filter(Filter<Predicate> && filter_obj,
                   OtherTransformers && ... other_transform_ops) const;

  template <typename Input, typename Combiner, typename Identity,
            template <typename C, typename I> class Reduce,
            typename ... OtherTransformers,
            requires_reduce<Reduce<Combiner,Identity>> = 0>
  auto make_filter(Reduce<Combiner,Identity> & reduce_obj,
                   OtherTransformers && ... other_transform_ops) const
  {
    return this->template make_filter<Input>(std::move(reduce_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename Input, typename Combiner, typename Identity,
            template <typename C, typename I> class Reduce,
            typename ... OtherTransformers,
            requires_reduce<Reduce<Combiner,Identity>> = 0>
  auto make_filter(Reduce<Combiner,Identity> && reduce_obj,
                   OtherTransformers && ... other_transform_ops) const;

  template <typename Input, typename Transformer, typename Predicate,
            template <typename T, typename P> class Iteration,
            typename ... OtherTransformers,
            requires_iteration<Iteration<Transformer,Predicate>> =0,
            requires_no_pattern<Transformer> =0>
  auto make_filter(Iteration<Transformer,Predicate> & iteration_obj,
                   OtherTransformers && ... other_transform_ops) const
  {
    return this->template make_filter<Input>(std::move(iteration_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename Input, typename Transformer, typename Predicate,
            template <typename T, typename P> class Iteration,
            typename ... OtherTransformers,
            requires_iteration<Iteration<Transformer,Predicate>> =0,
            requires_no_pattern<Transformer> =0>
  auto make_filter(Iteration<Transformer,Predicate> && iteration_obj,
                   OtherTransformers && ... other_transform_ops) const;

  template <typename Input, typename Transformer, typename Predicate,
            template <typename T, typename P> class Iteration,
            typename ... OtherTransformers,
            requires_iteration<Iteration<Transformer,Predicate>> =0,
            requires_pipeline<Transformer> =0>
  auto make_filter(Iteration<Transformer,Predicate> && iteration_obj,
                   OtherTransformers && ... other_transform_ops) const;

  template <typename Input, typename ... Transformers,
            template <typename...> class Pipeline,
            typename ... OtherTransformers,
            requires_pipeline<Pipeline<Transformers...>> = 0>
  auto make_filter(Pipeline<Transformers...> & pipeline_obj,
      OtherTransformers && ... other_transform_ops) const
  {
    return this->template make_filter<Input>(std::move(pipeline_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename Input, typename ... Transformers,
            template <typename...> class Pipeline,
            typename ... OtherTransformers,
            requires_pipeline<Pipeline<Transformers...>> = 0>
  auto make_filter(Pipeline<Transformers...> && pipeline_obj,
      OtherTransformers && ... other_transform_ops) const;

  template <typename Input, typename ... Transformers,
            std::size_t ... I>
  auto make_filter_nested(std::tuple<Transformers...> && transform_ops,
      std::index_sequence<I...>) const;

private:

  constexpr static int token_factor_ = 4;

  configuration<> config_{};

  int concurrency_degree_ = config_.concurrency_degree();

  bool ordering_ = true;

  int queue_size_ = config_.queue_size();

  int num_tokens_ = token_factor_ * concurrency_degree_;

  queue_mode queue_mode_ = config_.mode();
};

/**
\brief Metafunction that determines if type E is parallel_execution_tbb
\tparam Execution policy type.
*/
template <typename E>
constexpr bool is_parallel_execution_tbb() {
  return std::is_same<E, parallel_execution_tbb>::value;
}

/**
\brief Determines if an execution policy is supported in the current compilation.
\note Specialization for parallel_execution_omp when GRPPI_TBB is enabled.
*/
template <>
constexpr bool is_supported<parallel_execution_tbb>() { return true; }

/**
\brief Determines if an execution policy supports the map pattern.
\note Specialization for parallel_execution_omp when GRPPI_TBB is enabled.
*/
template <>
constexpr bool supports_map<parallel_execution_tbb>() { return true; }

/**
\brief Determines if an execution policy supports the reduce pattern.
\note Specialization for parallel_execution_omp when GRPPI_TBB is enabled.
*/
template <>
constexpr bool supports_reduce<parallel_execution_tbb>() { return true; }

/**
\brief Determines if an execution policy supports the map-reduce pattern.
\note Specialization for parallel_execution_omp when GRPPI_TBB is enabled.
*/
template <>
constexpr bool supports_map_reduce<parallel_execution_tbb>() { return true; }

/**
\brief Determines if an execution policy supports the stencil pattern.
\note Specialization for parallel_execution_omp when GRPPI_TBB is enabled.
*/
template <>
constexpr bool supports_stencil<parallel_execution_tbb>() { return true; }

/**
\brief Determines if an execution policy supports the divide/conquer pattern.
\note Specialization for parallel_execution_omp when GRPPI_TBB is enabled.
*/
template <>
constexpr bool supports_divide_conquer<parallel_execution_tbb>() { return true; }

/**
\brief Determines if an execution policy supports the pipeline pattern.
\note Specialization for parallel_execution_omp when GRPPI_TBB is enabled.
*/
template <>
constexpr bool supports_pipeline<parallel_execution_tbb>() { return true; }

/**
\brief Determines if an execution policy supports the stream pool pattern.
\note Specialization for parallel_execution_native.
*/
template <>
constexpr bool supports_stream_pool<parallel_execution_tbb>() { return true; }

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
        std::forward<Identity>(identity),
        std::forward<Transformer>(transform_op), 
        std::forward<Combiner>(combine_op));
  };

  const auto chunk_size = sequence_size/concurrency_degree_;

  for(int i=0; i<concurrency_degree_-1;++i) {    
    const auto delta = chunk_size * i;
    const auto chunk_firsts = iterators_next(firsts,delta);

    g.run([&, chunk_firsts, i]() {
      process_chunk(chunk_firsts, chunk_size, i);
    });
  }

  const auto delta = chunk_size * (concurrency_degree_ - 1);
  const auto chunk_firsts = iterators_next(firsts,delta);
  process_chunk(chunk_firsts, sequence_size - delta, concurrency_degree_-1);

  g.wait(); 

  return seq.reduce(partial_results.begin(), 
      partial_results.size(), std::forward<Identity>(identity), 
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
  const auto chunk_size = sequence_size / concurrency_degree_;
  auto process_chunk = [&](auto f, std::size_t sz, std::size_t delta) {
    constexpr sequential_execution seq{};
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

template <typename Input, typename Divider, typename Predicate, typename Solver, typename Combiner>
auto parallel_execution_tbb::divide_conquer(
    Input && input,
    Divider && divide_op,
    Predicate && predicate_op,
    Solver && solve_op,
    Combiner && combine_op) const
{
  std::atomic<int> num_threads{concurrency_degree_-1};
  return divide_conquer(std::forward<Input>(input),
        std::forward<Divider>(divide_op), std::forward<Predicate>(predicate_op),
        std::forward<Solver>(solve_op),
        std::forward<Combiner>(combine_op), num_threads);
}


template <typename Generator, typename ... Transformers>
void parallel_execution_tbb::pipeline(
    Generator && generate_op, 
    Transformers && ... transform_ops) const
{
  using namespace std;

  using result_type = decay_t<typename result_of<Generator()>::type>;
  using output_value_type = typename result_type::value_type;
  using output_type = grppi::optional<output_value_type>;

  auto generator = tbb::make_filter<void, output_type>(
    tbb::filter::serial_in_order, 
    [&](tbb::flow_control & fc) -> output_type {
      auto item =  generate_op();
      if (item) {
        return *item;
      }
      else {
        fc.stop();
        return {};
      }
    }
  );

  auto rest =
    this->template make_filter<output_value_type>(forward<Transformers>(transform_ops)...);


  tbb::task_group_context context;
  tbb::parallel_pipeline(tokens(), 
    generator
    & 
    rest);
}

template <typename Population, typename Selection, typename Evolution,
          typename Evaluation, typename Predicate>
void parallel_execution_tbb::stream_pool(Population & population,
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
  
  if( population.size() == 0 ) return;

  auto selected_queue = make_queue<selected_op_type>();
  auto output_queue = make_queue<individual_op_type>();

  std::atomic<bool> end{false};
  std::atomic<int> done_threads{0};
  std::atomic_flag lock = ATOMIC_FLAG_INIT;
  tbb::task_group g;
  
  for(auto i = 0; i< concurrency_degree_-2; i++){
    g.run( [&](){

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
    if(done_threads == concurrency_degree_-2){
      output_queue.push(individual_op_type{});
    }
    
   });
  }

  g.run([&](){
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
    for(int i=0;i<concurrency_degree_-2;i++){ 
      selected_queue.push(selected_op_type{});
    }
  });

  g.run([&](){
    auto item = output_queue.pop();
    while(item) {
      
      while(lock.test_and_set());
      population.push_back(*item);
      lock.clear();

      item = output_queue.pop();
    }
  });
  g.wait();
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

template <typename Input, typename Divider, typename Predicate, typename Solver, typename Combiner>
auto parallel_execution_tbb::divide_conquer(
    Input && input,
    Divider && divide_op,
    Predicate && predicate_op,
    Solver && solve_op,
    Combiner && combine_op,
    std::atomic<int> & num_threads) const
{
  constexpr sequential_execution seq;

  if (num_threads.load()<=0) {
    return seq.divide_conquer(std::forward<Input>(input),
        std::forward<Divider>(divide_op),std::forward<Predicate>(predicate_op),
        std::forward<Solver>(solve_op),
        std::forward<Combiner>(combine_op));
  }


  if (predicate_op(input)) { return solve_op(std::forward<Input>(input)); }
  auto subproblems = divide_op(std::forward<Input>(input));

  using subresult_type = std::decay_t<typename std::result_of<Solver(Input)>::type>;
  std::vector<subresult_type> partials(subproblems.size()-1);
  int division = 0;

  tbb::task_group g;
  auto i = subproblems.begin()+1;
  while (i!=subproblems.end() && num_threads.load()>0) {
    g.run([&,this,it=i++,div=division++]() {
        partials[div] = this->divide_conquer(std::forward<Input>(*it),
            std::forward<Divider>(divide_op), std::forward<Predicate>(predicate_op),
            std::forward<Solver>(solve_op),
            std::forward<Combiner>(combine_op), num_threads);
    });
    num_threads--;
  }

  //Main thread works on the first subproblem.
  while (i != subproblems.end()){
    partials[division] = seq.divide_conquer(std::forward<Input>(*i++),
        std::forward<Divider>(divide_op), std::forward<Predicate>(predicate_op),
        std::forward<Solver>(solve_op),
        std::forward<Combiner>(combine_op));
  }

  auto out = divide_conquer(std::forward<Input>(*subproblems.begin()),
      std::forward<Divider>(divide_op), std::forward<Predicate>(predicate_op), std::forward<Solver>(solve_op),
      std::forward<Combiner>(combine_op), num_threads);

  g.wait();

  return seq.reduce(partials.begin(), partials.size(),
      std::forward<subresult_type>(out), std::forward<Combiner>(combine_op));
}

template <typename Input, typename Transformer,
          requires_no_pattern<Transformer>>
auto parallel_execution_tbb::make_filter(
    Transformer && transform_op) const
{
  using namespace std;

  using input_value_type = Input; 
  using input_type = grppi::optional<input_value_type>;

  return tbb::make_filter<input_type, void>( 
      tbb::filter::serial_in_order, 
      [=](input_type item) {
          if (item) transform_op(*item);
      });
}

template <typename Input, typename Transformer, typename ... OtherTransformers,
          requires_no_pattern<Transformer>>
auto parallel_execution_tbb::make_filter(
    Transformer && transform_op,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;

  using input_value_type = Input; 
  static_assert(!is_void<input_value_type>::value, 
      "Transformer must take non-void argument");
  using input_type = grppi::optional<input_value_type>;
  using output_value_type = 
    decay_t<typename result_of<Transformer(input_value_type)>::type>;
  static_assert(!is_void<output_value_type>::value,
      "Transformer must return a non-void result");
  using output_type = grppi::optional<output_value_type>;


  return 
      tbb::make_filter<input_type, output_type>(
          tbb::filter::serial_in_order, 
          [=](input_type item) -> output_type {
              if (item) return transform_op(*item);
              else return {};
          })
    &
      this->template make_filter<output_value_type>(
          std::forward<OtherTransformers>(other_transform_ops)...);
}

template <typename Input, typename FarmTransformer,
          template <typename> class Farm,
          requires_farm<Farm<FarmTransformer>>>
auto parallel_execution_tbb::make_filter(
    Farm<FarmTransformer> && farm_obj) const
{
  using namespace std;

  using input_value_type = Input; 
  using input_type = grppi::optional<input_value_type>;

  return tbb::make_filter<input_type, void>(
      tbb::filter::parallel,
      [=](input_type item) {
        if (item) farm_obj(*item);
      });
}

template <typename Input, typename FarmTransformer, 
          template <typename> class Farm,
          typename ... OtherTransformers,
          requires_farm<Farm<FarmTransformer>>>
auto parallel_execution_tbb::make_filter(
    Farm<FarmTransformer> && farm_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;

  using input_value_type = Input;
  static_assert(!is_void<input_value_type>::value, 
      "Farm must take non-void argument");
  using input_type = grppi::optional<input_value_type>;
  using output_value_type = decay_t<typename result_of<FarmTransformer(input_value_type)>::type>;
  static_assert(!is_void<output_value_type>::value,
      "Farm must return a non-void result");
  using output_type = grppi::optional<output_value_type>;

  return tbb::make_filter<input_type, output_type>(
      tbb::filter::parallel,
      [&](input_type item) -> output_type {
        if (item) return farm_obj(*item);
        else return {};
      })
    &
      this->template make_filter<output_value_type>(
          std::forward<OtherTransformers>(other_transform_ops)...);
}

template <typename Input, typename Predicate,
          template <typename> class Filter,
          requires_filter<Filter<Predicate>>>
auto parallel_execution_tbb::make_filter(
    Filter<Predicate> &&) const
{
  using namespace std;

  using input_value_type = Input; 
  using input_type = grppi::optional<input_value_type>;

  return tbb::make_filter<input_type, void>(
      tbb::filter::parallel,
      [=](input_type item) {
        if (item) filter_obj(*item);
      });
}

template <typename Input, typename Predicate, 
          template <typename> class Filter,
          typename ... OtherTransformers,
          requires_filter<Filter<Predicate>>>
auto parallel_execution_tbb::make_filter(
    Filter<Predicate> && filter_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;

  using input_value_type = Input;
  static_assert(!is_void<input_value_type>::value, 
      "Filter must take non-void argument");
  using input_type = grppi::optional<input_value_type>;

  return tbb::make_filter<input_type, input_type>(
      tbb::filter::parallel,
      [&](input_type item) -> input_type {
        if (item && filter_obj(*item)) return item;
        else return {};
      })
    &
      this->template make_filter<input_value_type>(
          std::forward<OtherTransformers>(other_transform_ops)...);
}

template <typename Input, typename Combiner, typename Identity,
          template <typename C, typename I> class Reduce,
          typename ... OtherTransformers,
          requires_reduce<Reduce<Combiner,Identity>>>
auto parallel_execution_tbb::make_filter(
    Reduce<Combiner,Identity> && reduce_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;

  using input_value_type = Input;
  using input_type = grppi::optional<input_value_type>;

  return tbb::make_filter<input_type, input_type>(
      tbb::filter::serial,
      [&, it=std::vector<input_value_type>()](input_type item) -> input_type {
        if (!item) return {};
        reduce_obj.add_item(std::forward<Identity>(*item));
        if (reduce_obj.reduction_needed()) {
            constexpr sequential_execution seq;
            return reduce_obj.reduce_window(seq);
        }
        return {};
      })
    &
      this->template make_filter<input_value_type>(
          std::forward<OtherTransformers>(other_transform_ops)...);
}

template <typename Input, typename Transformer, typename Predicate,
          template <typename T, typename P> class Iteration,
          typename ... OtherTransformers,
          requires_iteration<Iteration<Transformer,Predicate>>,
          requires_no_pattern<Transformer>>
auto parallel_execution_tbb::make_filter(
    Iteration<Transformer,Predicate> && iteration_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;

  using input_value_type = Input;
  using input_type = grppi::optional<input_value_type>;

  return tbb::make_filter<input_type, input_type>(
      tbb::filter::serial,
      [&](input_type item) -> input_type {
        if (!item) return {};
        do {
          item = iteration_obj.transform(*item);
        } while (!iteration_obj.predicate(*item));
        return item;
      })
    &
      this->template make_filter<input_value_type>(
          std::forward<OtherTransformers>(other_transform_ops)...);
}

template <typename Input, typename Transformer, typename Predicate,
          template <typename T, typename P> class Iteration,
          typename ... OtherTransformers,
          requires_iteration<Iteration<Transformer,Predicate>>,
          requires_pipeline<Transformer>>
auto parallel_execution_tbb::make_filter(
    Iteration<Transformer,Predicate> &&,
    OtherTransformers && ...) const
{
  static_assert(!is_pipeline<Transformer>, "Not implemented");
}



template <typename Input, typename ... Transformers,
          template <typename...> class Pipeline,
          typename ... OtherTransformers,
          requires_pipeline<Pipeline<Transformers...>>>
auto parallel_execution_tbb::make_filter(
    Pipeline<Transformers...> && pipeline_obj,
    OtherTransformers && ... other_transform_ops) const
{
  return this->template make_filter_nested<Input>(
      std::tuple_cat(pipeline_obj.transformers(), 
          std::forward_as_tuple(other_transform_ops...)),
      std::make_index_sequence<sizeof...(Transformers)+sizeof...(OtherTransformers)>());
}

template <typename Input, typename ... Transformers,
          std::size_t ... I>
auto parallel_execution_tbb::make_filter_nested(
    std::tuple<Transformers...> && transform_ops,
    std::index_sequence<I...>) const
{
  return this->template make_filter<Input>(
      std::forward<Transformers>(std::get<I>(transform_ops))...);
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

} // end namespace grppi

#endif // GRPPI_TBB

#endif
