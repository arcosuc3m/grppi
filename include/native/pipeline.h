/**
* @version		GrPPI v0.2
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

#ifndef GRPPI_NATIVE_PIPELINE_H
#define GRPPI_NATIVE_PIPELINE_H

#include "parallel_execution_native.h"
#include "../common/pack_traits.h"
#include "../common/callable_traits.h"

#include <thread>
#include <experimental/optional>

namespace grppi{

/*
template <typename InQueue, typename OutQueue, int Index, 
          typename ... MoreTransformers,
          internal::requires_index_last<Index,MoreTransformers...> = 0>
void composed_pipeline(InQueue & input_queue, 
                       const pipeline_info<parallel_execution_native, MoreTransformers...> & pipe, 
                       OutQueue & output_queue, std::vector<std::thread> & tasks)
{
  composed_pipeline(pipe.exectype, input_queue, 
      std::get<Index>(pipe.stages), output_queue, tasks);
}


template <typename InQueue, typename OutQueue, int Index, 
          typename ... MoreTransformers,
          internal::requires_index_not_last<Index,MoreTransformers...> = 0>
void composed_pipeline(InQueue & input_queue, 
                       const pipeline_info<parallel_execution_native,MoreTransformers...> & pipeline_obj, 
                       OutQueue & output_queue, std::vector<std::thread> & tasks)
{
  using namespace std;
  using namespace experimental;

  using stage_type = 
      typename tuple_element<Index,decltype(pipeline_obj.stages)>::type;
  using input_type = typename InQueue::value_type;
  using input_value_type = typename input_type::value_type;
  using result_value_type = 
      typename result_of<stage_type(input_value_type)>::type;
  using result_type = optional<result_value_type>;

  parallel_execution_native & ex = pipeline_obj.exectype;
  static auto tmp_queue = ex.make_queue<result_type>();

  composed_pipeline(pipeline_obj.exectype, input_queue, 
      get<Index>(pipeline_obj.stages), tmp_queue, tasks);
  composed_pipeline<mpmc_queue<result_type>, 
      OutQueue, Index+1, MoreTransformers ...>(
          tmp_queue,pipeline_obj,output_queue,tasks);
}

template <typename InQueue, typename Transformer, typename OutQueue>
void composed_pipeline(parallel_execution_native & ex, InQueue & input_queue,
                       Transformer && transform_op, OutQueue & output_queue, 
                       std::vector<std::thread> & tasks)
{
  using namespace std;
  tasks.emplace_back([&]() {
    auto manager = ex.thread_manager();
    auto item = input_queue.pop();
    for (;;) {
      using output_type = typename OutQueue::value_type;
      if (!item) {
        output_queue.push(output_type{}); 
        break;
      }
      else {
        output_queue.push(transform_op(*item));
      }
      item = input_queue.pop();
    }
  });
}

//Last stage
template <typename InQueue, typename Consumer>
void pipeline_impl(parallel_execution_native & ex, InQueue& input_queue, 
                   Consumer && consume) 
{
  using namespace std;
  using input_type = typename InQueue::value_type;

  auto manager = ex.thread_manager();

  vector<input_type> elements;
  long current = 0;
  if (ex.is_ordered()){
    auto item = input_queue.pop();
    while (item.first) {
      if(current == item.second){
        consume(*item.first);
        current ++;
      }
      else {
        elements.push_back(item);
      }
      // TODO: Probably find_if() + erase 
      for (auto it=elements.begin(); it!=elements.end(); it++) {
        if(it->second == current) {
          consume(*it->first);
          elements.erase(it);
          current++;
          break;
        }
      }
      item = input_queue.pop( );
    }
    while (elements.size()>0) {
      // TODO: Probably find_if() + erase
      for (auto it = elements.begin(); it != elements.end(); it++) {
        if(it->second == current) {
          consume(*it->first);
          elements.erase(it);
          current++;
          break;
        }
      }
    }
  }
  else {
    auto item = input_queue.pop( );
    while (item.first) {
      consume(*item.first);
      item = input_queue.pop();
    }
  }
}

//Item reduce stage
template <typename Combiner, typename Identity, typename InQueue, typename ...MoreTransformers>
void pipeline_impl(parallel_execution_native & ex, InQueue & input_queue, 
                   reduction_info<parallel_execution_native, Combiner, Identity> & reduction_obj, MoreTransformers ...more_transform_ops) 
{
  using reduction_type = 
      reduction_info<parallel_execution_native,Combiner,Identity>;

  pipeline_impl(ex, input_queue, std::forward<reduction_type&&>(reduction_obj), std::forward<MoreTransformers...>(more_transform_ops...));
}

template <typename Combiner, typename Identity, typename InQueue, typename ...MoreTransformers>
void pipeline_impl(parallel_execution_native & ex, InQueue & input_queue,
                   reduction_info<parallel_execution_native, Combiner, Identity> && reduction_obj, MoreTransformers ...more_transform_ops)
{
  using reduction_type = 
      reduction_info<parallel_execution_native,Combiner,Identity>;
  pipeline_impl_ordered(ex, input_queue, std::forward<reduction_type>(reduction_obj), std::forward<MoreTransformers...>(more_transform_ops...));
}

template <typename Combiner, typename Identity, typename InQueue, typename ...MoreTransformers>
void pipeline_impl_ordered(parallel_execution_native & ex, InQueue & input_queue,
                   reduction_info<parallel_execution_native,Combiner,Identity> && reduction_obj, MoreTransformers ...more_transform_ops) 
{
  using namespace std;
  using namespace std::experimental;
  vector<thread> tasks;
  using input_type = typename InQueue::value_type;
  using input_value_type = typename input_type::first_type::value_type;
  
  using result_value_type = typename result_of<Combiner(input_value_type, input_value_type)>::type;
  using result_type = pair<optional<result_value_type>, long>;
  
  auto output_queue = ex.make_queue<result_type>();
  
  thread windower_task([&](){
    vector<input_value_type> values;
    long out_order=0;
    auto item {input_queue.pop()};
    for(;;){
      while (item.first && values.size() != reduction_obj.window_size) {
        values.push_back(*item.first);
        item = input_queue.pop();
      }
      if (values.size() > 0) {
        auto reduced_value = reduce(reduction_obj.exectype, values.begin(), values.end(), reduction_obj.identity,
            std::forward<Combiner>(reduction_obj.combine_op));
        output_queue.push({{reduced_value}, out_order});      
        out_order++;
        if (item.first) {
          if (reduction_obj.offset <= reduction_obj.window_size) {
            values.erase(values.begin(), values.begin() + reduction_obj.offset);
          }
          else {
            values.erase(values.begin(), values.end());
            auto diff = reduction_obj.offset - reduction_obj.window_size;
            while (diff > 0 && item.first) {
              item = input_queue.pop();
              diff--;
            }
          }
        }
      }
      if (!item.first) break;
    }
    output_queue.push({{},-1});
  });

  pipeline_impl(ex, output_queue, forward<MoreTransformers>(more_transform_ops) ... );
  windower_task.join();
}

//Filtering stage
template <typename Transformer, typename InQueue, typename... MoreTransformers>
void pipeline_impl(parallel_execution_native & ex, InQueue & input_queue,
                   filter_info<parallel_execution_native,Transformer> & filter_obj, 
                   MoreTransformers && ... more_transform_ops) 
{
  using filter_type = filter_info<parallel_execution_native,Transformer>;

  pipeline_impl(ex,input_queue, std::forward<filter_type>(filter_obj), 
    std::forward<MoreTransformers>(more_transform_ops)... );
}

template <typename Transformer, typename InQueue, typename... MoreTransformers>
void pipeline_impl_ordered(parallel_execution_native & ex, InQueue& input_queue,
                           filter_info<parallel_execution_native,Transformer> && filter_obj, 
                           MoreTransformers && ... more_transform_ops ) 
{
  using namespace std;
  vector<thread> tasks;

  using input_type = typename InQueue::value_type;
  using input_value_type = typename input_type::first_type;
  auto tmp_queue = ex.make_queue<input_type>();

  atomic<int> done_threads{0}; 
  for (int th=0; th<filter_obj.exectype.concurrency_degree(); th++) {
    tasks.emplace_back([&]() {
      auto manager = filter_obj.exectype.thread_manager();

      auto item{input_queue.pop()};
      while (item.first) {
        if (filter_obj.task(*item.first)) {
          tmp_queue.push(item);
        }
        else {
          tmp_queue.push(make_pair(input_value_type{},item.second) );
        } 
        item = input_queue.pop();
      }
      done_threads++;
      if (done_threads==filter_obj.exectype.concurrency_degree()) {
        tmp_queue.push(make_pair(input_value_type{}, -1));
      } 
      else {
        input_queue.push(item);
      }
    });
  }

  auto output_queue = ex.make_queue<input_type>();
  auto ordering_thread = thread{[&](){
    auto manager = ex.thread_manager();
    vector<input_type> elements;
    int current = 0;
    long order = 0;
    auto item{tmp_queue.pop()};
    for (;;) {
      if(!item.first && item.second == -1) break; 
      if (item.second == current) {
        if (item.first) {
          output_queue.push(make_pair(item.first,order));
          order++;
        }
        current++;
      }
      else {
        elements.push_back(item);
      }
      // TODO: Probably find_if() + erase 
      for (auto it=elements.begin(); it<elements.end(); it++) {
        if (it->second == current) {
          if (it->first) {
            output_queue.push(make_pair(it->first,order));
            order++;
          }
          elements.erase(it);
          current++;
          break;
        }
      }
      item = tmp_queue.pop();
    }
    while (elements.size()>0) {
      // TODO: Probably find_if() + erase 
      for (auto it=elements.begin(); it<elements.end(); it++) {
        if (it->second == current) {
          if(it->first) { 
            output_queue.push(make_pair(it->first,order));
            order++;
          }
          elements.erase(it);
          current++;
          break;
        }
      }
    }
    output_queue.push(item);
  }};

  pipeline_impl(ex, output_queue, forward<MoreTransformers>(more_transform_ops) ... );
  ordering_thread.join();
  for (auto && t : tasks) { t.join(); }
}

template <typename Transformer, typename InQueue, typename ... MoreTransformers>
void pipeline_impl_unordered(parallel_execution_native & ex, InQueue & input_queue,
                             filter_info<parallel_execution_native,Transformer> && filter_obj, 
                             MoreTransformers && ... more_transform_ops) 
{
  using namespace std;
  vector<thread> tasks;

  using input_type = typename InQueue::value_type;
  using input_value_type = typename input_type::first_type;
  auto output_queue = ex.make_queue<input_type>();

  atomic<int> done_threads{0};

  for (int th=0; th<filter_obj.exectype.concurrency_degree(); th++) {
    tasks.emplace_back([&]() {
      auto manager = filter_obj.exectype.thread_manager();

      auto item{input_queue.pop()};
      while (item.first) {
        if (filter_obj.task(*item.first)) { 
          output_queue.push(item);
        }
        item = input_queue.pop();
      }
      done_threads++;
      if (done_threads==filter_obj.exectype.concurrency_degree()) {
        output_queue.push( make_pair(input_value_type{}, -1) );
      }
      else {
        input_queue.push(item);
      }
    });
  }

  pipeline_impl(ex, output_queue, 
      forward<MoreTransformers>(more_transform_ops) ... );

  for (auto && t : tasks) { t.join(); }
}

template <typename Transformer, typename InQueue, typename ... MoreTransformers>
void pipeline_impl(parallel_execution_native & ex, InQueue& input_queue,
                   filter_info<parallel_execution_native,Transformer> && filter_obj, 
                   MoreTransformers && ... more_transform_ops) 
{
  using filter_type = filter_info<parallel_execution_native,Transformer>;
  if(ex.is_ordered()) {
    pipeline_impl_ordered(ex, input_queue, 
        std::forward<filter_type>(filter_obj),
        std::forward<MoreTransformers>(more_transform_ops)...);
  }
  else{
    pipeline_impl_unordered(ex, input_queue, 
        std::forward<filter_type>(filter_obj),
        std::forward<MoreTransformers>(more_transform_ops)...);
  }
}

template <typename Transformer, typename InQueue, typename... MoreTransformers>
void pipeline_impl(parallel_execution_native & ex, InQueue & input_queue,
                   farm_info<parallel_execution_native, Transformer> & farm_obj, 
                   MoreTransformers && ... more_transform_ops) 
{
  using farm_type = farm_info<parallel_execution_native,Transformer>;
  pipeline_impl(ex, input_queue, std::forward<farm_type>(farm_obj),
         std::forward< MoreTransformers>(more_transform_ops) ... );
}


//Farm stage
template <typename Transformer, typename InQueue, typename... MoreTransformers>
void pipeline_impl(parallel_execution_native & p, InQueue & input_queue, 
                   farm_info<parallel_execution_native,Transformer> && farm_obj, 
                   MoreTransformers && ... more_transform_ops) 
{
  using namespace std;

  using input_item_type = typename InQueue::value_type;
  using input_item_value_type = 
      typename input_item_type::first_type::value_type;
  using transform_result_type = 
      typename result_of<Transformer(input_item_value_type)>::type;
  using output_item_value_type = 
      experimental::optional<transform_result_type>;
  using output_item_type =
      pair<output_item_value_type,long>;
  auto output_queue = p.make_queue<output_item_type>();

  atomic<int> done_threads{0};
  vector<thread> tasks;
  for(int th = 0; th<farm_obj.exectype.concurrency_degree(); ++th){
    tasks.emplace_back([&]() {
      auto manager = farm_obj.exectype.thread_manager();

      long order = 0;
      auto item{input_queue.pop()}; 
      while (item.first) {
        auto out = output_item_value_type{farm_obj.task(*item.first)};
        output_queue.push(make_pair(out,item.second)) ;
        item = input_queue.pop( ); 
      }
      input_queue.push(item);
      done_threads++;
      if (done_threads == farm_obj.exectype.concurrency_degree()) {
        output_queue.push(make_pair(output_item_value_type{}, -1));
      }
    });
  }
  pipeline_impl(p, output_queue, 
      forward<MoreTransformers>(more_transform_ops)... );
    
  for (auto && t : tasks) { t.join(); }
}

//Intermediate pipeline_impl
template <typename Transformer, typename InQueue, typename... MoreTransformers>
void pipeline_impl(parallel_execution_native & ex, InQueue & input_queue, 
                   Transformer && transform_op, 
                   MoreTransformers && ... more_transform_ops) 
{
  using namespace std;

  using input_item_type = typename InQueue::value_type;
  using input_item_value_type = typename input_item_type::first_type::value_type;
  using transform_result_type = 
      typename result_of<Transformer(input_item_value_type)>::type;
  using output_item_value_type =
      experimental::optional<transform_result_type>;
  using output_item_type =
      pair<output_item_value_type,long>;

  auto output_queue = ex.make_queue<output_item_type>();

  thread task( 
    [&]() {
      auto manager = ex.thread_manager();

      long order = 0;
      auto item{input_queue.pop()};
      while(item.first) {
        auto out = output_item_value_type{transform_op(*item.first)};
        output_queue.push(make_pair(out, item.second));
        item = input_queue.pop( ) ;
      }
      output_queue.push(make_pair(output_item_value_type{},-1));
    }
  );

  pipeline_impl(ex, output_queue, 
      forward<MoreTransformers>(more_transform_ops)...);
  task.join();
}
*/

/**
\addtogroup pipeline_pattern
@{
\addtogroup pipeline_pattern_native Native parallel pipeline pattern
\brief Native parallel implementation of the \ref md_pipeline.
@{
*/

/**
\brief Invoke \ref md_pipeline on a data stream
with native parallel execution.
\tparam Generator Callable type for the stream generator.
\tparam Transformers Callable type for each transformation stage.
\param ex Native parallel execution policy object.
\param generate_op Generator operation.
\param trasnform_ops Transformation operations for each stage.
\remark Generator shall be a zero argument callable type.
*/
template <typename Generator, typename ... Transformers,
          requires_no_arguments<Generator> = 0>
void pipeline(parallel_execution_native & ex, Generator && generate_op, 
              Transformers && ... transform_ops) 
{
  ex.pipeline(std::forward<Generator>(generate_op),
      std::forward<Transformers>(transform_ops)...);
/*
  using namespace std;

  using result_type = typename result_of<Generator()>::type;
  using output_type = pair<result_type,long>;
  auto first_queue = ex.make_queue<output_type>();

  thread generator_task(
    [&]() {
      auto manager = ex.thread_manager();

      long order = 0;
      for (;;) {
        auto item{generate_op()};
        first_queue.push(make_pair(item, order));
        order++;
        if (!item) break;
      }
    }
  );

  pipeline_impl(ex, first_queue, forward<Transformers>(transform_ops)...);
  generator_task.join();
*/
}

/**
@}
@}
*/

}

#endif
