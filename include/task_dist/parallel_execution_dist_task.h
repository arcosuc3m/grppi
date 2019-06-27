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
#ifndef GRPPI_NATIVE_PARALLEL_EXECUTION_DIST_TASK_H
#define GRPPI_NATIVE_PARALLEL_EXECUTION_DIST_TASK_H

#include "pool.h"
#include "portService.h"
#include "dataService"

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


/** 
 \brief Native task-based parallel execution policy.
 This policy uses ISO C++ threads as implementation building block allowing
 usage in any ISO C++ compliant platform.
 */
template <typename Scheduler>
class parallel_execution_dist_task {
public:
  // Alias type for the scheduler and the task type
  using scheduler_type = Scheduler;
  using task_type = typename Scheduler::task_type;

  /** 
  \brief Default construct a task parallel execution policy.

  Creates a parallel execution task object.

  The concurrency degree is determined by the platform.

  \note The concurrency degree is fixed to the hardware concurrency
   degree.
  */
  parallel_execution_dist_task() noexcept:
    scheduler_{new Scheduler{}},
    thread_pool_{scheduler_,concurrency_degree_}
  {
  }
  parallel_execution_dist_task(int concurrency_degree) noexcept :
    concurrency_degree_{concurrency_degree},
    scheduler_{new Scheduler{}},
    thread_pool_{scheduler_,concurrency_degree_}
  {
  }
  parallel_execution_dist_task(int concurrency_degree, bool ordering) noexcept :
    concurrency_degree_{concurrency_degree},
    ordering_{ordering}
    scheduler_{new Scheduler{}},
    thread_pool{scheduler_,concurrency_degree_},
  {
  }

/**
  \brief Constructs a task parallel execution policy.

  Creates a parallel execution task object selecting the scheduler, concurrency degree
  and ordering mode.

  \param scheduler Scheduler to use.
  \param concurrency_degree Number of threads used for parallel algorithms.
  \param ordering Whether ordered executions is enabled or disabled.
  */
  parallel_execution_dist_task(std::shared_ptr<Scheduler> scheduler) noexcept :
    scheduler_{scheduler},
    thread_pool{scheduler_,concurrency_degree_},
  {
  }
  parallel_execution_dist_task(std::shared_ptr<Scheduler> scheduler,
                               int concurrency_degree) noexcept :
    concurrency_degree_{concurrency_degree},
    scheduler_{scheduler},
    thread_pool_{scheduler_,concurrency_degree_}
  {
  }
  parallel_execution_dist_task(std::shared_ptr<Scheduler> scheduler,
                               int concurrency_degree, bool ordering) noexcept :
    concurrency_degree_{concurrency_degree},
    ordering_{ordering}
    scheduler_{scheduler},
    thread_pool{scheduler_,concurrency_degree_},
  {
  }

  parallel_execution_dist_task(Scheduler scheduler, int concurrency_degree) noexcept :
    parallel_execution_dist_task(scheduler
  
  parallel_execution_dist_task(const parallel_execution_dist_task & ex) = delete;

  /** 
  \brief Destroy a task parallel execution policy.

  */
  ~parallel_execution_dist_task(){
     thread_pool_.finalize_pool();
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
  \brief Invoke \ref md_pipeline.
  \tparam Generator Callable type for the generator operation.
  \tparam Transformers Callable types for the transformers in the pipeline.
  \param generate_op Generator operation.
  \param transform_ops Transformer operations.
  */
  template <typename Generator, typename ... Transformers>
  void pipeline(Generator && generate_op, 
                Transformers && ... transform_ops) const;

//  /**
//  \brief Invoke \ref md_pipeline coming from another context
//  that uses mpmc_queues as communication channels.
//  \tparam InputType Type of the input stream.
//  \tparam Transformers Callable types for the transformers in the pipeline.
//  \tparam InputType Type of the output stream.
//  \param input_queue Input stream communicator.
//  \param transform_ops Transformer operations.
//  \param output_queue Input stream communicator.
//  */
//  template <typename InputType, typename Transformer, typename OutputType>
//  void pipeline(std::shared_ptr<mpmc_queue<InputType>> & input_queue, Transformer && transform_op,
//                std::shared_ptr<mpmc_queue<OutputType>> &output_queue) const
//  {
//    do_pipeline(input_queue, std::forward<Transformer>(transform_op), output_queue);
//  }

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

  template <typename Consumer,
            requires_no_pattern<Consumer> = 0>
  void do_pipeline(Consumer && consume_op) const;

  template <typename Transformer, typename ... OtherTransformers,
            requires_no_pattern<Transformer> = 0>
  void do_pipeline(Transformer && transform_op,
      OtherTransformers && ... other_ops) const;
    
    
//  template <typename Inqueue, typename Transformer, typename output_type,
//            requires_no_pattern<Transformer> = 0>
//  void do_pipeline(Inqueue & input_queue, Transformer && transform_op,
//      std::shared_ptr<mpmc_queue<output_type>> & output_queue) const;
//
//  template <typename T, typename ... Others>
//  void do_pipeline(std::shared_ptr<mpmc_queue<T>> & in_q, std::shared_ptr<mpmc_queue<T>> & same_queue, Others &&... ops) const;
//
//  template <typename T>
//  void do_pipeline(std::shared_ptr<mpmc_queue<T>> &) const{}
//
//  template <typename Queue, typename FarmTransformer,
//            template <typename> class Farm,
//            requires_farm<Farm<FarmTransformer>> = 0>
//  void do_pipeline(Queue & input_queue,
//      Farm<FarmTransformer> & farm_obj) const
//  {
//    do_pipeline(input_queue, std::move(farm_obj));
//  }
//
//  template <typename Queue, typename FarmTransformer,
//            template <typename> class Farm,
//            requires_farm<Farm<FarmTransformer>> = 0>
//  void do_pipeline( Queue & input_queue,
//      Farm<FarmTransformer> && farm_obj) const;
//
//  template <typename Queue, typename Execution, typename Transformer,
//            template <typename, typename> class Context,
//            typename ... OtherTransformers,
//            requires_context<Context<Execution,Transformer>> = 0>
//  void do_pipeline(Queue & input_queue, Context<Execution,Transformer> && context_op,
//       OtherTransformers &&... other_ops) const;
//
//  template <typename Queue, typename Execution, typename Transformer,
//            template <typename, typename> class Context,
//            typename ... OtherTransformers,
//            requires_context<Context<Execution,Transformer>> = 0>
//  void do_pipeline(Queue & input_queue, Context<Execution,Transformer> & context_op,
//       OtherTransformers &&... other_ops) const
//  {
//    do_pipeline(input_queue, std::move(context_op),
//      std::forward<OtherTransformers>(other_ops)...);
//  }
//
//  template <typename Queue, typename FarmTransformer,
//            template <typename> class Farm,
//            typename ... OtherTransformers,
//            requires_farm<Farm<FarmTransformer>> = 0>
//  void do_pipeline(Queue & input_queue,
//      Farm<FarmTransformer> & farm_obj,
//      OtherTransformers && ... other_transform_ops) const
//  {
//    do_pipeline(input_queue, std::move(farm_obj),
//        std::forward<OtherTransformers>(other_transform_ops)...);
//  }
//
//  template <typename Queue, typename FarmTransformer,
//            template <typename> class Farm,
//            typename ... OtherTransformers,
//            requires_farm<Farm<FarmTransformer>> = 0>
//  void do_pipeline(Queue & input_queue,
//      Farm<FarmTransformer> && farm_obj,
//      OtherTransformers && ... other_transform_ops) const;
//
//  template <typename Queue, typename Predicate,
//            template <typename> class Filter,
//            typename ... OtherTransformers,
//            requires_filter<Filter<Predicate>> =0>
//  void do_pipeline(Queue & input_queue,
//      Filter<Predicate> & filter_obj,
//      OtherTransformers && ... other_transform_ops) const
//  {
//    do_pipeline(input_queue, std::move(filter_obj),
//        std::forward<OtherTransformers>(other_transform_ops)...);
//  }
//
//  template <typename Queue, typename Predicate,
//            template <typename> class Filter,
//            typename ... OtherTransformers,
//            requires_filter<Filter<Predicate>> =0>
//  void do_pipeline(Queue & input_queue,
//      Filter<Predicate> && farm_obj,
//      OtherTransformers && ... other_transform_ops) const;
//
//  template <typename Queue, typename Combiner, typename Identity,
//            template <typename C, typename I> class Reduce,
//            typename ... OtherTransformers,
//            requires_reduce<Reduce<Combiner,Identity>> = 0>
//  void do_pipeline(Queue && input_queue, Reduce<Combiner,Identity> & reduce_obj,
//                   OtherTransformers && ... other_transform_ops) const
//  {
//    do_pipeline(input_queue, std::move(reduce_obj),
//        std::forward<OtherTransformers>(other_transform_ops)...);
//  }
//
//  template <typename Queue, typename Combiner, typename Identity,
//            template <typename C, typename I> class Reduce,
//            typename ... OtherTransformers,
//            requires_reduce<Reduce<Combiner,Identity>> = 0>
//  void do_pipeline(Queue && input_queue, Reduce<Combiner,Identity> && reduce_obj,
//                   OtherTransformers && ... other_transform_ops) const;
//
//  template <typename Queue, typename Transformer, typename Predicate,
//            template <typename T, typename P> class Iteration,
//            typename ... OtherTransformers,
//            requires_iteration<Iteration<Transformer,Predicate>> =0,
//            requires_no_pattern<Transformer> =0>
//  void do_pipeline(Queue & input_queue, Iteration<Transformer,Predicate> & iteration_obj,
//                   OtherTransformers && ... other_transform_ops) const
//  {
//    do_pipeline(input_queue, std::move(iteration_obj),
//        std::forward<OtherTransformers>(other_transform_ops)...);
//  }
//
//  template <typename Queue, typename Transformer, typename Predicate,
//            template <typename T, typename P> class Iteration,
//            typename ... OtherTransformers,
//            requires_iteration<Iteration<Transformer,Predicate>> =0,
//            requires_no_pattern<Transformer> =0>
//  void do_pipeline(Queue & input_queue, Iteration<Transformer,Predicate> && iteration_obj,
//                   OtherTransformers && ... other_transform_ops) const;
//
//  template <typename Queue, typename Transformer, typename Predicate,
//            template <typename T, typename P> class Iteration,
//            typename ... OtherTransformers,
//            requires_iteration<Iteration<Transformer,Predicate>> =0,
//            requires_pipeline<Transformer> =0>
//  void do_pipeline(Queue & input_queue, Iteration<Transformer,Predicate> && iteration_obj,
//                   OtherTransformers && ... other_transform_ops) const;
//
//
//  template <typename Queue, typename ... Transformers,
//            template <typename...> class Pipeline,
//            requires_pipeline<Pipeline<Transformers...>> = 0>
//  void do_pipeline(Queue & input_queue,
//      Pipeline<Transformers...> & pipeline_obj) const
//  {
//    do_pipeline(input_queue, std::move(pipeline_obj));
//  }
//
//  template <typename Queue, typename ... Transformers,
//            template <typename...> class Pipeline,
//            requires_pipeline<Pipeline<Transformers...>> = 0>
//  void do_pipeline(Queue & input_queue,
//      Pipeline<Transformers...> && pipeline_obj) const;
//
//  template <typename Queue, typename ... Transformers,
//            template <typename...> class Pipeline,
//            typename ... OtherTransformers,
//            requires_pipeline<Pipeline<Transformers...>> = 0>
//  void do_pipeline(Queue & input_queue,
//      Pipeline<Transformers...> & pipeline_obj,
//      OtherTransformers && ... other_transform_ops) const
//  {
//    do_pipeline(input_queue, std::move(pipeline_obj),
//        std::forward<OtherTransformers>(other_transform_ops)...);
//  }
//
//  template <typename Queue, typename ... Transformers,
//            template <typename...> class Pipeline,
//            typename ... OtherTransformers,
//            requires_pipeline<Pipeline<Transformers...>> = 0>
//  void do_pipeline(Queue & input_queue,
//      Pipeline<Transformers...> && pipeline_obj,
//      OtherTransformers && ... other_transform_ops) const;
//
//  template <typename Queue, typename ... Transformers,
//            std::size_t ... I>
//  void do_pipeline_nested(
//      Queue & input_queue,
//      std::tuple<Transformers...> && transform_ops,
//      std::index_sequence<I...>) const;

private: 

  configuration<> config_{};

  mutable std::shared_ptr<Scheduler> scheduler_;

  mutable pool<Scheduler> thread_pool_;
  
  configuration<> config_{};
  
  int concurrency_degree_ = config_.concurrency_degree();

  bool ordering_ = config_.ordering();
 
};

/**
\brief Metafunction that determines if type E is parallel_execution_dist_task
\tparam Execution policy type.
*/
template <typename E,typename T>
constexpr bool is_parallel_execution_dist_task() {
  return std::is_same<E, parallel_execution_dist_task<T> >::value;
}

template <typename T>
using execution_task = parallel_execution_dist_task<T>;

/**
\brief Determines if an execution policy is supported in the current compilation.
\note Specialization for parallel_execution_dist_task.
*/
template <typename U>
struct support<parallel_execution_dist_task<U>>
{
   static constexpr bool value = true;
};

/*template <>
constexpr bool is_supported<execution_task>() { return true; }
*/
/**
\brief Determines if an execution policy supports the map pattern.
\note Specialization for parallel_execution_dist_task.
*/
/*template <>
constexpr bool supports_map<execution_task>() { return true; }
*/
/**
\brief Determines if an execution policy supports the reduce pattern.
\note Specialization for parallel_execution_dist_task.
*/
/*template <>
constexpr bool supports_reduce<parallel_execution_dist_task>() { return true; }
*/
/**
\brief Determines if an execution policy supports the map-reduce pattern.
\note Specialization for parallel_execution_dist_task.
*/
/*template <>
constexpr bool supports_map_reduce<parallel_execution_dist_task>() { return true; }
*/
/**
\brief Determines if an execution policy supports the stencil pattern.
\note Specialization for parallel_execution_dist_task.
*/
/*template <>
constexpr bool supports_stencil<parallel_execution_dist_task>() { return true; }
*/
/**
\brief Determines if an execution policy supports the divide/conquer pattern.
\note Specialization for parallel_execution_dist_task.
*/
/*template <>
constexpr bool supports_divide_conquer<parallel_execution_dist_task>() { return true; }
*/
/**
\brief Determines if an execution policy supports the pipeline pattern.
\note Specialization for parallel_execution_dist_task.
*/
/*template <>
constexpr bool supports_pipeline<parallel_execution_dist_task>() { return true; }
*/

template <typename Scheduler>
template <typename Generator, typename ... Transformers>
void parallel_execution_dist_task<Scheduler>::pipeline(
    Generator && generate_op, 
    Transformers && ... transform_ops) const
{
  using namespace std;
  using result_type = decay_t<typename result_of<Generator()>::type>;
  using output_type = pair<result_type,long>;

  long order=0;
  (void) scheduler_.register_parallel_stage([&generate_op, this, &order](task_type t){
     auto item{generate_op()};
     if(item){ 
       scheduler_.new_token();
       auto ref = scheduler_->set(make_pair(item, order));
       thread_pool_.launch_task(t);
       thread_pool_.launch_task(task_type{1,0,ref});
       order++;
     } else {
       scheduler_.pipe_stop();
     }
  });

  do_pipeline<output_type>(forward<Transformers>(transform_ops)...);
}
// PRIVATE MEMBERS
template <typename Scheduler>
template <typename InputItemType, typename Consumer,
          requires_no_pattern<Consumer>>
void parallel_execution_dist_task<Scheduler>::do_pipeline(
    Consumer && consume_op) const
{
  using namespace std;
  //TODO: Need to reimplement ordering
    (void) scheduler_.register_sequential_task([&consume_op, this](task_type t){
       auto item = scheduler_->get<InputItemType>(t.get_data_location());
       consume_op(*item.first);
       scheduler_.notify_consumer_end();
       scheduler_.notify_sequential_end(t);
    });
  scheduler_.run();
}

template <typename Scheduler>
template <typename InputItemType, typename Transformer,
          typename ... OtherTransformers,
          requires_no_pattern<Transformer>>
void parallel_execution_dist_task<Scheduler>::do_pipeline(
    Transformer && transform_op,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
  using namespace experimental;

  using input_item_value_type = typename InputItemType::first_type::value_type;
  using transform_result_type = 
      decay_t<typename result_of<Transformer(input_item_value_type)>::type>;
  using output_item_value_type = optional<transform_result_type>;
  using output_item_type = pair<output_item_value_type,long>;

  (void) scheduler_.register_sequential_task([this,&transform_op](task_type t) {
    auto item = scheduler_->get<InputItemType>(t.get_data_location());
    auto out = output_item_value_type{transform_op(*item.first)};
    auto ref = scheduler_->set(make_pair(out,item.second));
    scheduler_.launch_task(task_type{t.get_id()+1,0,ref});
    scheduler_.notify_sequential_end(t);
  });

  do_pipeline<output_item_type>(forward<OtherTransformers>(other_transform_ops)...);
}

template <typename Scheduler>
template <typename InputItemType, typename Transformer, typename OutputItemType
            requires_no_pattern<Transformer>>
void parallel_execution_dist_task<Scheduler>::do_pipeline(Transformer && transform_op) const
{
  using namespace std;
  using namespace experimental;

  using output_item_value_type = typename OutputItemType::first_type::value_type;
  (void) scheduler_.register_parallel_stage([this, &transform_op](task_type t){
    auto item = scheduler_->get<InputItemType>(t.get_data_location());
    auto out = output_item_value_type{transform_op(*item.first)};
    auto ref = scheduler_->set(make_pair(out,item.second));
    scheduler_.launch_task(task_type{t.get_id()+1,0,ref});
  });
}

template <typename Scheduler>
template <typename InputItemType, typename FarmTransformer,
          template <typename> class Farm,
          requires_farm<Farm<FarmTransformer>>>
void parallel_execution_dist_task<Scheduler>::do_pipeline(
    Farm<FarmTransformer> && farm_obj) const
{
  using namespace std;

  (void)scheduler_.register_parallel_stage([this,&farm_obj](task_type t){
    auto item = scheduler_->get<InputItemType>(t.get_data_location());
    farm_obj(*item.first);
    scheduler_.notify_consumer_end();
  });

  scheduler_.run();

}

template <typename Scheduler>
template <typename InputItemType, typename FarmTransformer,
          template <typename> class Farm,
          typename ... OtherTransformers,
          requires_farm<Farm<FarmTransformer>>>
void parallel_execution_dist_task<Scheduler>::do_pipeline(
    Farm<FarmTransformer> && farm_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
  using namespace experimental;

  using input_item_value_type = typename InputItemType::first_type::value_type;

  using output_type = typename stage_return_type<input_item_value_type, FarmTransformer>::type;
  using output_optional_type = optional<output_type>;
  using output_item_type = pair <output_optional_type, long> ;

  do_pipeline<InputItemType,output_item_type>(farm_obj.transformer());
  do_pipeline<output_item_type>(forward<OtherTransformers>(other_transform_ops)... );
}

template <typename Scheduler>
template <typename InputItemType, typename Predicate,
          template <typename> class Filter,
          typename ... OtherTransformers,
          requires_filter<Filter<Predicate>>>
void parallel_execution_dist_task<Scheduler>::do_pipeline(
    Filter<Predicate> && filter_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
  using namespace experimental;

  (void) scheduler_.register_parallel_stage([&filter_obj, this](task_type t){
      auto item = scheduler_->get<InputItemType>(t.get_data_location());
      if (filter_obj(*item.first)) {
        auto ref = scheduler_->set(item);
        scheduler_.launch_task(task_type{t.get_id()+1,0,ref});
      }
      else {
        scheduler_.notify_consumer_end();
      }
  });

  do_pipeline<InputItemType>(forward<OtherTransformers>(other_transform_ops)...);

}

template <typename Scheduler>
template <typename InputItemType, typename Combiner, typename Identity,
          template <typename C, typename I> class Reduce,
          typename ... OtherTransformers,
          requires_reduce<Reduce<Combiner,Identity>>>
void parallel_execution_dist_task<Scheduler>::do_pipeline(
    Reduce<Combiner,Identity> && reduce_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
  using namespace experimental;

  using output_item_value_type = optional<decay_t<Identity>>;
  using output_item_type = pair<output_item_value_type,long>;

  // Review if it can be transformed into parallel task
  // Transform into atomic if used as a parallel task
  long int order = 0;

  scheduler_.register_sequential_task([&reduce_obj, this, &order](task_type t){
    auto item = scheduler_->get<InputItemType>(t.get_data_location());
    reduce_obj.add_item(std::forward<Identity>(*item.first));
    if(reduce_obj.reduction_needed()) {
      constexpr sequential_execution seq;
      auto red = reduce_obj.reduce_window(seq);
      auto ref = scheduler_->set(make_pair(red, order++));
      scheduler_.launch_task(task_type{t.get_id()+1,0,ref});
    } else{
      scheduler_.notify_consumer_end();
    }
    scheduler_.notify_sequential_end(t);
  });

  do_pipeline<output_item_type>(forward<OtherTransformers>(other_transform_ops)...);
}

template <typename Scheduler>
template <typename InputItemType, typename Transformer, typename Predicate,
          template <typename T, typename P> class Iteration,
          typename ... OtherTransformers,
          requires_iteration<Iteration<Transformer,Predicate>>,
          requires_no_pattern<Transformer>>
void parallel_execution_dist_task<Scheduler>::do_pipeline(
    Iteration<Transformer,Predicate> && iteration_obj,
    OtherTransformers && ... other_transform_ops) const
{
  using namespace std;
  using namespace experimental;

  (void) scheduler_.register_parallel_stage([&iteration_obj, this](task_type t){
      auto item = scheduler_->get<InputItemType>(t.get_data_location());
      auto value = iteration_obj.transform(*item.first);
      auto new_item = input_item_type{value,item.second};
      if (iteration_obj.predicate(value)) {
        auto ref = scheduler_->set(new_item);
        scheduler_.launch_task(task_type{t.get_id()+1, 0, ref} );
      }
      else {
        auto ref = scheduler_->set(new_item);
        t.set_data_location(ref);
        scheduler_.launch_task(t);
      }
  });

  do_pipeline<InputItemType>(forward<OtherTransformers>(other_transform_ops)...);
}

template <typename Scheduler>
template <typename InputItemType, typename Transformer, typename Predicate,
          template <typename T, typename P> class Iteration,
          typename ... OtherTransformers,
          requires_iteration<Iteration<Transformer,Predicate>>,
          requires_pipeline<Transformer>>
void parallel_execution_dist_task<Scheduler>::do_pipeline(
    Iteration<Transformer,Predicate> &&,
    OtherTransformers && ...) const
{
  static_assert(!is_pipeline<Transformer>, "Not implemented");
}

template <typename Scheduler>
template <typename InputItemType, typename ... Transformers,
          template <typename...> class Pipeline,
          requires_pipeline<Pipeline<Transformers...>>>
void parallel_execution_dist_task<Scheduler>::do_pipeline(
    Pipeline<Transformers...> && pipeline_obj) const
{
  do_pipeline_nested<InputItemType>(
      pipeline_obj.transformers(),
      std::make_index_sequence<sizeof...(Transformers)>());
}

template <typename Scheduler>
template <typename InputItemType, typename ... Transformers,
          template <typename...> class Pipeline,
          typename ... OtherTransformers,
          requires_pipeline<Pipeline<Transformers...>>>
void parallel_execution_dist_task<Scheduler>::do_pipeline(
    Pipeline<Transformers...> && pipeline_obj,
    OtherTransformers && ... other_transform_ops) const
{
  do_pipeline_nested<InputItemType>(
      std::tuple_cat(pipeline_obj.transformers(),
          std::forward_as_tuple(other_transform_ops...)),
      std::make_index_sequence<sizeof...(Transformers)+sizeof...(OtherTransformers)>());
}

template <typename Scheduler>
template <typename InputItemType, typename ... Transformers,
          std::size_t ... I>
void parallel_execution_dist_task<Scheduler>::do_pipeline_nested(
    std::tuple<Transformers...> && transform_ops,
    std::index_sequence<I...>) const
{
  do_pipeline<InputItemType>(input_queue,
      std::forward<Transformers>(std::get<I>(transform_ops))...);
}

//template <typename Scheduler>
//template<typename T, typename... Others>
//void parallel_execution_dist_task<Scheduler>::do_pipeline(std::shared_ptr<mpmc_queue<T>> &, std::shared_ptr<mpmc_queue<T>> &, Others &&...) const
//{
//}

//template <typename Scheduler>
//template <typename Queue, typename Execution, typename Transformer,
//          template <typename, typename> class Context,
//          typename ... OtherTransformers,
//          requires_context<Context<Execution,Transformer>>>
//void parallel_execution_dist_task<Scheduler>::do_pipeline(Queue & input_queue,
//    Context<Execution,Transformer> && context_op,
//    OtherTransformers &&... other_ops) const
//{
//  // WARNING: Ignore context - Inner context do not launch the next task.
//  do_pipeline(input_queue, std::forward<Transformer>(context_op.transformer()),
//    std::forward<OtherTransformers>(other_ops)...);
//
//  /*using namespace std;
//  using namespace experimental;
//
//  using input_item_type = typename Queue::value_type;
//  using input_item_value_type = typename input_item_type::first_type::value_type;
//
//  using output_type = typename stage_return_type<input_item_value_type, Transformer>::type;
//  using output_optional_type = optional<output_type>;
//  using output_item_type = pair <output_optional_type, long> ;
//
//  decltype(auto) output_queue =
//    get_output_queue<output_item_type>(other_ops...);
//
//
//  auto context_task = [&]() {
//    context_op.execution_policy().pipeline(input_queue, context_op.transformer(), output_queue);
//    output_queue.push( make_pair(output_optional_type{}, -1) );
//  };
//
//  do_pipeline(input_queue, std::forward<Transformer>(context_op.transformer()),
//      forward<OtherTransformers>(other_ops)... );
//
//  do_pipeline(input_queue, context_op.transformer(), output_queue);
//  do_pipeline(output_queue,
//      forward<OtherTransformers>(other_ops)... );
//*/
//}
//

} // end namespace grppi

#endif
