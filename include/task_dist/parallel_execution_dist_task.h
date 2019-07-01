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

#include "dist_pool.h"
#include "zmq_port_service.h"
#include "zmq_data_service.h"

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
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstring>

#include <boost/serialization/utility.hpp>

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
    config_{},
    concurrency_degree_{config_.concurrency_degree()},
    ordering_{config_.ordering()},
    scheduler_{new Scheduler{}},
    thread_pool_{scheduler_,concurrency_degree_}
  {
  }
  parallel_execution_dist_task(int concurrency_degree) noexcept :
    config_{},
    concurrency_degree_{concurrency_degree},
    ordering_{config_.ordering()},
    scheduler_{new Scheduler{}},
    thread_pool_{scheduler_,concurrency_degree_}
  {
  }
  parallel_execution_dist_task(int concurrency_degree, bool ordering) noexcept :
    config_{},
    concurrency_degree_{concurrency_degree},
    ordering_{ordering},
    scheduler_{new Scheduler{}},
    thread_pool_{scheduler_,concurrency_degree_}
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
    config_{},
    concurrency_degree_{config_.concurrency_degree()},
    ordering_{config_.ordering()},
    scheduler_{scheduler},
    thread_pool_{scheduler_,concurrency_degree_}
  {
    //std::cout<<"parallel_execution_dist_task: concurrency_degree_ = " << concurrency_degree_ << std::endl;
  }
  parallel_execution_dist_task(std::shared_ptr<Scheduler> scheduler,
                               int concurrency_degree) noexcept :
    config_{},
    concurrency_degree_{concurrency_degree},
    ordering_{config_.ordering()},
    scheduler_{scheduler},
    thread_pool_{scheduler_,concurrency_degree_}
  {
  }
  parallel_execution_dist_task(std::shared_ptr<Scheduler> scheduler,
                               int concurrency_degree, bool ordering) noexcept :
    config_{},
    concurrency_degree_{concurrency_degree},
    ordering_{ordering},
    scheduler_{scheduler},
    thread_pool_{scheduler_,concurrency_degree_}
  {
  }

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

  template <typename InputItemType, typename Consumer,
            requires_no_pattern<Consumer> = 0>
  void do_pipeline(Consumer && consume_op) const;

  template <typename InputItemType, typename Transformer,
            typename ... OtherTransformers,
            requires_no_pattern<Transformer> = 0>
  void do_pipeline(Transformer && transform_op,
      OtherTransformers && ... other_ops) const;
    
    
  template <typename InputItemType, typename Transformer,
            requires_no_pattern<Transformer> = 0>
  void do_pipeline(Transformer && transform_op, bool check) const;


  template <typename InputItemType, typename FarmTransformer,
            template <typename> class Farm,
            requires_farm<Farm<FarmTransformer>> = 0>
  void do_pipeline(Farm<FarmTransformer> & farm_obj) const
  {
    do_pipeline<InputItemType>(std::move(farm_obj));
  }

  template <typename InputItemType, typename FarmTransformer,
            template <typename> class Farm,
            requires_farm<Farm<FarmTransformer>> = 0>
  void do_pipeline(Farm<FarmTransformer> && farm_obj) const;

  template <typename InputItemType, typename FarmTransformer,
            template <typename> class Farm,
            typename ... OtherTransformers,
            requires_farm<Farm<FarmTransformer>> = 0>
  void do_pipeline(
      Farm<FarmTransformer> & farm_obj,
      OtherTransformers && ... other_transform_ops) const
  {
    do_pipeline<InputItemType>(std::move(farm_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename InputItemType, typename FarmTransformer,
            template <typename> class Farm,
            typename ... OtherTransformers,
            requires_farm<Farm<FarmTransformer>> = 0>
  void do_pipeline(
      Farm<FarmTransformer> && farm_obj,
      OtherTransformers && ... other_transform_ops) const;

  template <typename InputItemType, typename Predicate,
            template <typename> class Filter,
            typename ... OtherTransformers,
            requires_filter<Filter<Predicate>> =0>
  void do_pipeline(
      Filter<Predicate> & filter_obj,
      OtherTransformers && ... other_transform_ops) const
  {
    do_pipeline<InputItemType>(std::move(filter_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename InputItemType, typename Predicate,
            template <typename> class Filter,
            typename ... OtherTransformers,
            requires_filter<Filter<Predicate>> =0>
  void do_pipeline(
      Filter<Predicate> && farm_obj,
      OtherTransformers && ... other_transform_ops) const;

  template <typename InputItemType, typename Combiner, typename Identity,
            template <typename C, typename I> class Reduce,
            typename ... OtherTransformers,
            requires_reduce<Reduce<Combiner,Identity>> = 0>
  void do_pipeline(Reduce<Combiner,Identity> & reduce_obj,
                   OtherTransformers && ... other_transform_ops) const
  {
    do_pipeline<InputItemType>(std::move(reduce_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename InputItemType, typename Combiner, typename Identity,
            template <typename C, typename I> class Reduce,
            typename ... OtherTransformers,
            requires_reduce<Reduce<Combiner,Identity>> = 0>
  void do_pipeline(Reduce<Combiner,Identity> && reduce_obj,
                   OtherTransformers && ... other_transform_ops) const;

  template <typename InputItemType, typename Transformer, typename Predicate,
            template <typename T, typename P> class Iteration,
            typename ... OtherTransformers,
            requires_iteration<Iteration<Transformer,Predicate>> =0,
            requires_no_pattern<Transformer> =0>
  void do_pipeline(Iteration<Transformer,Predicate> & iteration_obj,
                   OtherTransformers && ... other_transform_ops) const
  {
    do_pipeline<InputItemType>(std::move(iteration_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename InputItemType, typename Transformer, typename Predicate,
            template <typename T, typename P> class Iteration,
            typename ... OtherTransformers,
            requires_iteration<Iteration<Transformer,Predicate>> =0,
            requires_no_pattern<Transformer> =0>
  void do_pipeline(Iteration<Transformer,Predicate> && iteration_obj,
                   OtherTransformers && ... other_transform_ops) const;

  template <typename InputItemType, typename Transformer, typename Predicate,
            template <typename T, typename P> class Iteration,
            typename ... OtherTransformers,
            requires_iteration<Iteration<Transformer,Predicate>> =0,
            requires_pipeline<Transformer> =0>
  void do_pipeline(Iteration<Transformer,Predicate> && iteration_obj,
                   OtherTransformers && ... other_transform_ops) const;


  template <typename InputItemType, typename ... Transformers,
            template <typename...> class Pipeline,
            requires_pipeline<Pipeline<Transformers...>> = 0>
  void do_pipeline(
      Pipeline<Transformers...> & pipeline_obj) const
  {
    do_pipeline<InputItemType>(std::move(pipeline_obj));
  }

  template <typename InputItemType, typename ... Transformers,
            template <typename...> class Pipeline,
            requires_pipeline<Pipeline<Transformers...>> = 0>
  void do_pipeline(
      Pipeline<Transformers...> && pipeline_obj) const;

  template <typename InputItemType, typename ... Transformers,
            template <typename...> class Pipeline,
            typename ... OtherTransformers,
            requires_pipeline<Pipeline<Transformers...>> = 0>
  void do_pipeline(
      Pipeline<Transformers...> & pipeline_obj,
      OtherTransformers && ... other_transform_ops) const
  {
    do_pipeline<InputItemType>(std::move(pipeline_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename InputItemType, typename ... Transformers,
            template <typename...> class Pipeline,
            typename ... OtherTransformers,
            requires_pipeline<Pipeline<Transformers...>> = 0>
  void do_pipeline(
      Pipeline<Transformers...> && pipeline_obj,
      OtherTransformers && ... other_transform_ops) const;

  template <typename InputItemType, typename ... Transformers,
            std::size_t ... I>
  void do_pipeline_nested(
      std::tuple<Transformers...> && transform_ops,
      std::index_sequence<I...>) const;

private: 
  configuration<> config_;
  
  int concurrency_degree_;

  bool ordering_;

  mutable std::shared_ptr<Scheduler> scheduler_;

  mutable dist_pool<Scheduler> thread_pool_;
   
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
using execution_dist_task = parallel_execution_dist_task<T>;

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
  using output_type = pair<typename result_type::value_type,long>;

  //std::cout << "pipeline: generator" << std::endl;
  long order=0;
  std::cout << "GENERATOR" << std::endl;
  (void) scheduler_->register_parallel_stage([&generate_op, this, &order](task_type t){
     {std::ostringstream foo;
     foo << "task["<< t.get_id() << ","<< t.get_task_id()<< "]: generator, ref=(" << t.get_data_location().get_id() << "," << t.get_data_location().get_pos() << ")" << std::endl;
     std::cout << foo.str();}
     auto item{generate_op()};
     if(item){
       //std::cout << "task: generator item = true" << std::endl;
       scheduler_->new_token();
       //std::cout << "task: generator set begin" << std::endl;
       auto ref = scheduler_->set(make_pair(*item, order));
       //std::cout << "task: generator set end" << std::endl;
       {std::ostringstream foo;
       foo << "task["<< t.get_id() << ","<< t.get_task_id()<< "]: generator, launch task[" << t.get_id()+1 <<"," << order << "] ref=(" << ref.get_id() << "," << ref.get_pos() << ")" << std::endl;
       std::cout << foo.str();}
       thread_pool_.launch_task(task_type{t.get_id()+1,order,ref});
       // increase order
       order++;
       {std::ostringstream foo;
       foo << "task["<< t.get_id() << ","<< t.get_task_id()<< "]: generator, launch task[" << t.get_id() << ","<< order << "], ref=(" << t.get_data_location().get_id() << "," << t.get_data_location().get_pos() << ")" << std::endl;
       std::cout << foo.str();}
       thread_pool_.launch_task(task_type{t.get_id(),order});
     } else {
       //std::cout << "task: generator item = false" << std::endl;
       scheduler_->pipe_stop();
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
  //std::cout << "pipeline: consumer" << std::endl;
  using namespace std;
  //TODO: Need to reimplement ordering
    std::cout << "CONSUMER" << std::endl;
    (void) scheduler_->register_sequential_task([&consume_op, this](task_type t){
       {std::ostringstream foo;
       foo << "task["<< t.get_id() << ","<< t.get_task_id()<< "]: consumer, ref=(" << t.get_data_location().get_id() << "," << t.get_data_location().get_pos() << ")" << std::endl;
       std::cout << foo.str();}
       auto item = scheduler_->template get<InputItemType>(t.get_data_location());
       consume_op(item.first);
       scheduler_->notify_consumer_end();
       scheduler_->notify_sequential_end(t);
    });
  scheduler_->run();
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

  using input_item_value_type = typename InputItemType::first_type;
  using transform_result_type = 
      decay_t<typename result_of<Transformer(input_item_value_type)>::type>;
  using output_item_type = pair<transform_result_type,long>;

  std::cout << "NO PATTERN" << std::endl;
  (void) scheduler_->register_sequential_task([this,&transform_op](task_type t) {
    {std::ostringstream foo;
    foo << "task["<< t.get_id() << ","<< t.get_task_id()<< "]: no_pattern, ref=(" << t.get_data_location().get_id() << "," << t.get_data_location().get_pos() << ")" << std::endl;
    std::cout << foo.str();}
    auto item = scheduler_->template get<InputItemType>(t.get_data_location());
    auto out = transform_op(item.first);
    auto ref = scheduler_->set(make_pair(out,item.second));
    {std::ostringstream foo;
    foo << "task["<< t.get_id() << ","<< t.get_task_id()<< "]: no_pattern, launch task[" << t.get_id()+1 <<"," << t.get_task_id() << "] ref=(" << ref.get_id() << "," << ref.get_pos() << ")" << std::endl;
    std::cout << foo.str();}
    scheduler_->launch_task(task_type{t.get_id()+1,t.get_task_id(),ref});
    scheduler_->notify_sequential_end(t);
  });

  do_pipeline<output_item_type>(forward<OtherTransformers>(other_transform_ops)...);
}

template <typename Scheduler>
template <typename InputItemType, typename Transformer,
            requires_no_pattern<Transformer>>
void parallel_execution_dist_task<Scheduler>::do_pipeline(Transformer && transform_op, bool check) const
{
  using namespace std;
  using namespace experimental;
  
  if (!check) {
    return;
  }
  
  std::cout << "NO PATTERN END" << std::endl;
  
  (void) scheduler_->register_parallel_stage([this, &transform_op](task_type t){
    {std::ostringstream foo;
    foo << "task["<< t.get_id() << ","<< t.get_task_id()<< "]: no_pattern_farm, ref=(" << t.get_data_location().get_id() << "," << t.get_data_location().get_pos() << ")" << std::endl;
    std::cout << foo.str();}
    auto item = scheduler_->template get<InputItemType>(t.get_data_location());
    auto out = transform_op(item.first);
    auto ref = scheduler_->set(make_pair(out,item.second));
    {std::ostringstream foo;
    foo << "task["<< t.get_id() << ","<< t.get_task_id()<< "]: no_pattern_farm, launch task[" << t.get_id()+1 <<"," << t.get_task_id() << "] ref=(" << ref.get_id() << "," << ref.get_pos() << ")" << std::endl;
    std::cout << foo.str();}
    scheduler_->launch_task(task_type{t.get_id()+1,t.get_task_id(),ref});
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

  std::cout << "FARM CONSUMER" << std::endl;
  (void)scheduler_->register_parallel_stage([this,&farm_obj](task_type t){
    {std::ostringstream foo;
    foo << "task["<< t.get_id() << ","<< t.get_task_id()<< "]: farm consumer, ref=(" << t.get_data_location().get_id() << "," << t.get_data_location().get_pos() << ")" << std::endl;
    std::cout << foo.str();}
    auto item = scheduler_->template get<InputItemType>(t.get_data_location());
    farm_obj(item.first);
    scheduler_->notify_consumer_end();
  });

  scheduler_->run();

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

  using input_item_value_type = typename InputItemType::first_type;

  using output_type = typename stage_return_type<input_item_value_type, FarmTransformer>::type;
  using output_item_type = pair <output_type, long> ;

  std::cout << "FARM" << std::endl;
  do_pipeline<InputItemType>(farm_obj.transformer(),true);
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

  std::cout << "FILTER" << std::endl;
  (void) scheduler_->register_parallel_stage([&filter_obj, this](task_type t){
      {std::ostringstream foo;
      foo << "task["<< t.get_id() << ","<< t.get_task_id()<< "]: filter, ref=(" << t.get_data_location().get_id() << "," << t.get_data_location().get_pos() << ")" << std::endl;
      std::cout << foo.str();}
      auto item = scheduler_->template get<InputItemType>(t.get_data_location());
      if (filter_obj(item.first)) {
        auto ref = scheduler_->set(item);
        {std::ostringstream foo;
        foo << "task["<< t.get_id() << ","<< t.get_task_id()<< "]: filter, launch task[" << t.get_id()+1 <<"," << t.get_task_id() << "] ref=(" << ref.get_id() << "," << ref.get_pos() << ")" << std::endl;
        std::cout << foo.str();}
        scheduler_->launch_task(task_type{t.get_id()+1,t.get_task_id(),ref});
      } else {
        {std::ostringstream foo;
        foo << "task["<< t.get_id() << ","<< t.get_task_id()<< "]: filter is consumed" << std::endl;
        std::cout << foo.str();}
        scheduler_->notify_consumer_end();
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

  using output_item_value_type = decay_t<Identity>;
  using output_item_type = pair<output_item_value_type,long>;

  // Review if it can be transformed into parallel task
  // Transform into atomic if used as a parallel task
  long int order = 0;

  std::cout << "REDUCE" << std::endl;
  scheduler_->register_sequential_task([&reduce_obj, this, &order](task_type t){
    {std::ostringstream foo;
    foo << "task["<< t.get_id() << ","<< t.get_task_id()<< "]: reduce, ref=(" << t.get_data_location().get_id() << "," << t.get_data_location().get_pos() << ")" << std::endl;
    std::cout << foo.str();}
    auto item = scheduler_->template get<InputItemType>(t.get_data_location());
    reduce_obj.add_item(std::forward<Identity>(item.first));
    if(reduce_obj.reduction_needed()) {
      constexpr sequential_execution seq;
      auto red = reduce_obj.reduce_window(seq);
      auto ref = scheduler_->set(make_pair(red, order++));
      {std::ostringstream foo;
      foo << "task["<< t.get_id() << ","<< t.get_task_id()<< "]: reduce, launch task[" << t.get_id()+1 <<"," << t.get_task_id() << "] ref=(" << ref.get_id() << "," << ref.get_pos() << ")" << std::endl;
      std::cout << foo.str();}
      scheduler_->launch_task(task_type{t.get_id()+1,t.get_task_id(),ref});
    } else{
      scheduler_->notify_consumer_end();
    }
    scheduler_->notify_sequential_end(t);
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

  std::cout << "ITERATION" << std::endl;
  (void) scheduler_->register_parallel_stage([&iteration_obj, this](task_type t){
      {std::ostringstream foo;
      foo << "task["<< t.get_id() << ","<< t.get_task_id()<< "]: iteration, ref=(" << t.get_data_location().get_id() << "," << t.get_data_location().get_pos() << ")" << std::endl;
      std::cout << foo.str();}
      auto item = scheduler_->template get<InputItemType>(t.get_data_location());
      auto value = iteration_obj.transform(item.first);
      auto new_item = InputItemType{value,item.second};
      if (iteration_obj.predicate(value)) {
        auto ref = scheduler_->set(new_item);
        {std::ostringstream foo;
        foo << "task["<< t.get_id() << ","<< t.get_task_id()<< "]: iteration, launch task[" << t.get_id()+1 <<"," << t.get_task_id() << "] ref=(" << ref.get_id() << "," << ref.get_pos() << ")" << std::endl;
        std::cout << foo.str();}
        scheduler_->launch_task(task_type{t.get_id()+1, t.get_task_id(), ref} );
      }
      else {
        auto ref = scheduler_->set(new_item);
        t.set_data_location(ref);
        {std::ostringstream foo;
        foo << "task["<< t.get_id() << ","<< t.get_task_id()<< "]: iteration, launch task[" << t.get_id() <<"," << t.get_task_id() << "] ref=(" << ref.get_id() << "," << ref.get_pos() << ")" << std::endl;
        std::cout << foo.str();}
        scheduler_->launch_task(t);
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
  std::cout << "PIPELINE 1" << std::endl;
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
  std::cout << "PIPELINE 1-1" << std::endl;
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
  std::cout << "PIPELINE 2" << std::endl;
  do_pipeline<InputItemType>(std::forward<Transformers>(std::get<I>(transform_ops))...);
}

} // end namespace grppi

#endif
