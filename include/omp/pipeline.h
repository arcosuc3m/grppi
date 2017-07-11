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

#ifndef GRPPI_PIPELINE_OMP_H
#define GRPPI_PIPELINE_OMP_H

#ifdef GRPPI_OMP

#include <experimental/optional>

#include <boost/lockfree/spsc_queue.hpp>

namespace grppi{

//Last stage
template <typename InQueue, typename Consumer>
void pipeline_impl(parallel_execution_omp & ex, InQueue & input_queue, 
                   Consumer && consume_op)
{
  using namespace std;
  using input_type = typename InQueue::value_type;

  if (ex.ordering){
    vector<input_type> elements;
    long current = 0;
    auto item = input_queue.pop( );
    while (item.first) {
      if (current == item.second) {
        consume_op( item.first.value() );
        current ++;
      } 
      else {
        elements.push_back(item);
      }
      for (auto it=elements.begin(); it!=elements.end(); it++) {
        if (it->second == current) {
          consume_op(*it->first);
              elements.erase(it);
              current++;
              break;
           }
        }
       item = input_queue.pop( );
      }
      while(elements.size()>0){
        for(auto it = elements.begin(); it != elements.end(); it++){
          if(it->second == current) {
            consume_op(*it->first);
            elements.erase(it);
            current++;
            break;
          }
        }
      }
    }
    else {
      auto item = input_queue.pop();
      while (item.first) {
        consume_op(*item.first);
        item = input_queue.pop();
     }
   }
   //End task
}

template <typename Transformer, typename InQueue, typename... MoreTransformers>
void pipeline_impl(parallel_execution_omp & ex, InQueue& input_queue, 
                   filter_info<parallel_execution_omp,Transformer> & filter_obj, 
                   MoreTransformers && ... more_transform_ops) 
{
  pipeline_impl(ex,input_queue,
      std::forward<filter_info<parallel_execution_omp, Transformer>>(filter_obj), 
      std::forward<MoreTransformers>(more_transform_ops)...) ;
}

template <typename Transformer, typename InQueue,
          typename... MoreTransformers>
void pipeline_impl_ordered(parallel_execution_omp & ex, 
                           InQueue & input_queue, 
                           filter_info<parallel_execution_omp,Transformer> && filter_obj, 
                           MoreTransformers && ... more_transform_ops)
{
  using namespace std;
  using input_type = typename InQueue::value_type;
  using input_value_type = typename input_type::first_type;
  mpmc_queue<input_type> tmp_queue{ex.queue_size, ex.lockfree};

  atomic<int> nend{0}; // TODO: Find better name?
  for(int th = 0; th<filter_obj.exectype.num_threads; th++) {
    #pragma omp task shared(tmp_queue,filter_obj,input_queue,nend)
    {
      auto item{input_queue.pop()};
      while (item.first) {
        if(filter_obj.task(*item.first)) {
          tmp_queue.push(item);
        }
        else {
          tmp_queue.push(make_pair(input_value_type{} ,item.second));
        }
        item = input_queue.pop();
      }
      nend++;
      if (nend==filter_obj.exectype.num_threads) {
        tmp_queue.push (make_pair(input_value_type{}, -1));
      }
      else {
        input_queue.push(item);
      }
    }
  }

  mpmc_queue<input_type> output_queue{ex.queue_size, ex.lockfree};
  #pragma omp task shared (output_queue,tmp_queue)
  {
    vector<input_type> elements;
    int current = 0;
    long order = 0;
    auto item = tmp_queue.pop();
    for (;;) {
      if (!item.first && item.second == -1) break;
      if (item.second == current) {
        if (item.first) {
          output_queue.push(make_pair(item.first, order++));
        }
        current++;
      }
      else {
        elements.push_back(item);
      }
      for (auto it=elements.begin(); it<elements.end(); it++) {
        if ((*it).second==current) {
          if((*it).first){
            output_queue.push(make_pair((*it).first,order++));
          }
          elements.erase(it);
          current++;
          break;
        }
      }
      item = tmp_queue.pop();
    }
    while (elements.size()>0) {
      for (auto it=elements.begin(); it<elements.end(); it++) {
        if ((*it).second == current) {
          if((*it).first) {
            output_queue.push(make_pair((*it).first,order++));
          }
          elements.erase(it);
          current++;
          break;
        }
      }
    }
    output_queue.push(item);
  }
  pipeline_impl(ex, output_queue, 
      forward<MoreTransformers>(more_transform_ops)...);
  #pragma omp taskwait
}

template <typename Transformer, typename InQueue,typename... MoreTransformers>
void pipeline_impl_unordered(parallel_execution_omp & ex, InQueue & input_queue, 
                             filter_info<parallel_execution_omp, Transformer> && farm_obj, 
                             MoreTransformers && ... more_transform_ops)
{
  using input_type = typename InQueue::value_type;
  using input_value_type = typename input_type::first_type;
  mpmc_queue<input_type> output_queue{ex.queue_size, ex.lockfree};

  std::atomic<int> nend{0}; //TODO: Better naming?
  for (int th=0; th<farm_obj.exectype.num_threads; th++) {
    #pragma omp task shared(output_queue,farm_obj,input_queue,nend)
    {
      auto item = input_queue.pop( ) ;
      while (item.first) {
        if (farm_obj.task(item.first.value())) {
          output_queue.push(item);
        }
        item = input_queue.pop();
      }
      nend++;
      if (nend==farm_obj.exectype.num_threads) {
        output_queue.push(make_pair(input_value_type{}, -1));
      }
      else {
        input_queue.push(item);
      }
    }
  }
  pipeline_impl(ex, output_queue, 
      std::forward<MoreTransformers>(more_transform_ops)...);
  #pragma omp taskwait
}

template <typename Transformer, typename InQueue,typename... MoreTransformers>
void pipeline_impl(parallel_execution_omp & ex, InQueue & input_queue, 
                   filter_info<parallel_execution_omp, Transformer> && filter_obj, 
                   MoreTransformers && ... more_transform_ops) 
{
  if (ex.ordering) {
    pipeline_impl_ordered(ex, input_queue,
        std::forward<filter_info<parallel_execution_omp, Transformer>>(filter_obj),
        std::forward<MoreTransformers>(more_transform_ops)...);
  }
  else {
    pipeline_impl_unordered(ex, input_queue,
        std::forward<filter_info<parallel_execution_omp, Transformer>>(filter_obj),
        std::forward<MoreTransformers>(more_transform_ops)...);
  }
}


template <typename Transformer, typename InQueue,typename... MoreTransformers>
void pipeline_impl(parallel_execution_omp & ex, InQueue & input_queue, 
                   farm_info<parallel_execution_omp, Transformer> & farm_obj, 
                   MoreTransformers && ... more_transform_ops) 
{
  pipeline_impl(ex, input_queue, 
      std::forward< farm_info<parallel_execution_omp,Transformer>>(farm_obj), 
      std::forward<MoreTransformers>(more_transform_ops)...) ;
}

template <typename Transformer, typename InQueue,typename... MoreTransformers>
void pipeline_impl(parallel_execution_omp & ex, InQueue & input_queue, 
                   farm_info<parallel_execution_omp, Transformer> && farm_obj, 
                   MoreTransformers && ... sgs ) 
{
  using namespace std;
  using input_type = typename InQueue::value_type;
  using input_value_type = typename input_type::first_type::value_type;
  using result_type = typename result_of<Transformer(input_value_type)>::type;
  using output_value_type = experimental::optional<result_type>;
  using output_type = pair<output_value_type,long>;
 
  mpmc_queue<output_type> output_queue{ex.queue_size, ex.lockfree};
  atomic<int> nend{0}; // TODO: Better name?
  for (int th=0; th<farm_obj.exectype.num_threads; th++) {
    #pragma omp task shared(nend,output_queue,farm_obj,input_queue)
    {
      auto item = input_queue.pop();
      while (item.first) {
        auto out = output_value_type{farm_obj.task(*item.first)};
        output_queue.push(make_pair(out,item.second));
        item = input_queue.pop();
      }
      input_queue.push(item);
      nend++;
      if (nend==farm_obj.exectype.num_threads) {
        output_queue.push(make_pair(output_value_type{}, -1));
      }
    }              
  }
  pipeline_impl(ex, output_queue, forward<MoreTransformers>(sgs) ... );
  #pragma omp taskwait
}

//Intermediate stages
template <typename Transformer, typename InQueue,typename ... MoreTransformers>
void pipeline_impl(parallel_execution_omp & ex, InQueue & input_queue, 
                   Transformer && transform_op, 
                   MoreTransformers && ... more_transform_ops) 
{
  using namespace std;
  using input_type = typename InQueue::value_type;
  using input_value_type = typename input_type::first_type::value_type;
  using result_type = typename result_of<Transformer(input_value_type)>::type;
  using output_value_type = experimental::optional<result_type>;
  using output_type = pair<output_value_type,long>;
  mpmc_queue<output_type> output_queue{ex.queue_size, ex.lockfree};

  //Start task
  #pragma omp task shared(transform_op, input_queue, output_queue)
  {
    auto item = input_queue.pop(); 
    while (item.first) {
      auto out = output_value_type{transform_op(*item.first)};
      output_queue.push(make_pair(out, item.second));
      item = input_queue.pop() ;
    }
    output_queue.push(make_pair(output_value_type{}, -1));
  }
  //End task

  pipeline_impl(ex, output_queue, 
      forward<MoreTransformers>(more_transform_ops)...);
}

/**
\addtogroup pipeline_pattern
@{
*/

/**
\addtogroup pipeline_pattern_omp OpenMP parallel pipeline pattern
\brief OpenMP parallel implementation of the \ref md_pipeline pattern
@{
*/

/**
\brief Invoke [pipeline pattern](@ref md_pipeline) on a data stream
with OpenMP parallel execution.
\tparam Generator Callable type for the stream generator.
\tparam MoreTransformers Callable type for each transformation stage.
\param ex Sequential execution policy object.
\param generate_op Generator operation.
\param trasnform_ops Transformation operations for each stage.
\remark Generator shall be a zero argument callable type.
*/
template <typename Generator, typename ... MoreTransformers,
          requires_no_arguments<Generator> = 0>
void pipeline(parallel_execution_omp & ex, Generator && generate_op, 
              MoreTransformers && ... more_transform_ops) 
{
  using namespace std;

  using result_type = typename result_of<Generator()>::type;
  mpmc_queue<pair<result_type,long>> output_queue{ex.queue_size,ex.lockfree};

  #pragma omp parallel
  {
    #pragma omp single nowait
    {
      #pragma omp task shared(generate_op,output_queue)
      {
        long order = 0;
        for (;;) {
          auto item = generate_op();
          output_queue.push(make_pair(item,order++)) ;
          if (!item) break;
        }
      }
      pipeline_impl(ex, output_queue,
          forward<MoreTransformers>(more_transform_ops)...);
      #pragma omp taskwait
    }
  }
}

/**
@}
@}
*/

}
#endif

#endif
