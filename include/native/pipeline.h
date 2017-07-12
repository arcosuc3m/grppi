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

#include "common/pack_traits.h"

#include <experimental/optional>

#include <thread>


namespace grppi{

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

  using stage_type = 
      typename tuple_element<Index,decltype(pipeline_obj.stages)>::type;
  using input_type = typename InQueue::value_type;
  using input_value_type = typename input_type::value_type;
  using result_value_type = 
      typename result_of<stage_type(input_value_type)>::type;
  using result_type = experimental::optional<result_value_type>;

  // TODO: Why static?
  static mpmc_queue<result_type> tmp_queue{
      pipeline_obj.exectype.queue_size, pipeline_obj.exectype.lockfree}; 

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
  tasks.push_back(
    thread{[&](){
      ex.register_thread();
        
      auto item = input_queue.pop();
      for (;;) {
        using output_type = typename OutQueue::value_type;
        using output_value_type = typename output_type::value_type;
        if (!item) {
          output_queue.push(output_value_type{}); 
          break;
        }
        else {
          auto out = output_value_type{transform_op(item.value())};
          output_queue.push(out);
        }
        item = input_queue.pop();
      }

      ex.deregister_thread();
    }}
  );
}

//Last stage
template <typename InQueue, typename Consumer>
void pipeline_impl(parallel_execution_native & ex, InQueue& input_queue, 
                   Consumer && consume) 
{
  using namespace std;
  using input_type = typename InQueue::value_type;

  ex.register_thread();

  vector<input_type> elements;
  long current = 0;
  if (ex.ordering){
    auto item = input_queue.pop();
    while (item.first) {
      if(current == item.second){
        consume(item.first.value());
        current ++;
      }
      else {
        elements.push_back(item);
      }
      // TODO: Probably find_if() + erase 
      for (auto it=elements.begin(); it!=elements.end(); it++) {
        if(it->second == current) {
          consume(it->first.value());
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
          consume(it->first.value());
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
      consume(item.first.value());
      item = input_queue.pop();
    }
  }
  
  ex.deregister_thread();
}

//Item reduce stage
template <typename Transformer, typename Reducer, typename InQueue>
void pipeline_impl(parallel_execution_native & ex, InQueue & input_queue,
                   reduction_info<parallel_execution_native, Transformer, Reducer> & reduction_obj) 
{
  using reduction_type = 
      reduction_info<parallel_execution_native,Transformer,Reducer>;

  pipeline_impl(ex, input_queue, std::forward<reduction_type>(reduction_obj));
}



template <typename Transformer, typename Reducer, typename InQueue>
void pipeline_impl(parallel_execution_native & ex, InQueue & input_queue,
                   reduction_info<parallel_execution_native,Transformer,Reducer> && reduction_obj) 
{
  using namespace std;
  vector<thread> tasks;
  using input_type = typename InQueue::value_type;
  using result_type = typename result_of<Transformer(input_type)>::type;
  mpmc_queue<result_type> output_queue{ex.queue_size,ex.lockfree};

  for (int th=0; th<reduction_obj.exectype.num_threads; th++) {
    tasks.push_back(
      thread{[&](){
        auto item = input_queue.pop( );
        while (item) {
          auto local =  input_queue.task(item) ;
          output_queue.push( local ) ;
          item = input_queue.pop( );
        }
        output_queue.push(result_type{}) ;
      }}
    );
  }
  // TODO: Remove commented line?
  //pipeline_impl(p, q, sgs ... );

  for (auto && t : tasks) { t.join(); }
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
  mpmc_queue<input_type> tmp_queue{ex.queue_size, ex.lockfree};

  atomic<int> nend{0}; // TODO: Find better name?
  for (int th=0; th<filter_obj.exectype.num_threads; th++) {
    tasks.push_back(
      thread{[&](){
        filter_obj.exectype.register_thread();

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
        nend++;
        if (nend==filter_obj.exectype.num_threads) {
          tmp_queue.push(make_pair(input_value_type{}, -1));
        } 
        else {
          input_queue.push(item);
        }

        filter_obj.exectype.deregister_thread();
      }}
    );
  }

  mpmc_queue<input_type> output_queue{ex.queue_size,ex.lockfree};
  auto ordering_thread = thread{[&](){
    ex.register_thread();
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
    ex.deregister_thread();
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
  mpmc_queue<input_type> output_queue{ex.queue_size, ex.lockfree};

  atomic<int> nend{0}; // TODO: Find better name?

  for (int th=0; th<filter_obj.exectype.num_threads; th++) {
    tasks.push_back(
      thread{[&]() {
        filter_obj.exectype.register_thread();

        auto item{input_queue.pop()};
        while (item.first) {
          if (filter_obj.task(*item.first)) { 
            output_queue.push(item);
          }
//        else { // TODO: Remove?
//          output_queue.push(
//              make_pair(typename input_value_type(), item.second));
//        } 
          item = input_queue.pop();
        }
        nend++;
        if (nend==filter_obj.exectype.num_threads) {
          output_queue.push( make_pair(input_value_type{}, -1) );
        }
        else {
          input_queue.push(item);
        }

        filter_obj.exectype.deregister_thread();
      }}
    );
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
  if(ex.ordering) {
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
  mpmc_queue<output_item_type> output_queue{p.queue_size,p.lockfree};

  atomic<int> nend{0}; //TODO: Look for better name
  vector<thread> tasks;
  for(int th = 0; th<farm_obj.exectype.num_threads; ++th){
    tasks.push_back(
      thread{[&]() {
        farm_obj.exectype.register_thread();

        long order = 0;
        auto item{input_queue.pop()}; 
        while (item.first) {
          auto out = output_item_value_type{farm_obj.task(*item.first)};
          output_queue.push(make_pair(out,item.second)) ;
          item = input_queue.pop( ); 
        }
        input_queue.push(item);
        nend++;
        if (nend == farm_obj.exectype.num_threads) {
          output_queue.push(make_pair(output_item_value_type{}, -1));
        }
                
        farm_obj.exectype.deregister_thread();
      }}
    );
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

  mpmc_queue<output_item_type> output_queue{ex.queue_size,ex.lockfree};

  thread task( 
    [&]() {
      ex.register_thread();

      long order = 0;
      auto item{input_queue.pop()};
      while(item.first) {
        auto out = output_item_value_type{transform_op(*item.first)};
        output_queue.push(make_pair(out, item.second));
        item = input_queue.pop( ) ;
      }
      output_queue.push(make_pair(output_item_value_type{},-1));

      ex.deregister_thread();
    }
  );

  pipeline_impl(ex, output_queue, 
      forward<MoreTransformers>(more_transform_ops)...);
  task.join();
}

/**
\addtogroup pipeline_pattern
@{
*/

/**
\addtogroup pipeline_pattern_native Native parallel pipeline pattern
\brief Native parallel implementation of the \ref md_pipeline pattern
@{
*/

/**
\brief Invoke [pipeline pattern](@ref md_pipeline) on a data stream
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
  using namespace std;

  using result_type = typename result_of<Generator()>::type;
  using output_type = pair<result_type,long>;
  mpmc_queue<output_type> first_queue{ex.queue_size,ex.lockfree};

  thread generator_task(
    [&]() {
      ex.register_thread();

      long order = 0;
      for (;;) {
        auto item{generate_op()};
        first_queue.push(make_pair(item, order));
        order++;
        if (!item) break;
      }

      ex.deregister_thread();
    }
  );

  pipeline_impl(ex, first_queue, forward<Transformers>(transform_ops)...);
  generator_task.join();
}

/**
@}
@}
*/

}

#endif
