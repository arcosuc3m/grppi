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

#ifndef GRPPI_TBB_STREAM_FILTER_H
#define GRPPI_TBB_STREAM_FILTER_H

#ifdef GRPPI_TBB

#include <tbb/tbb.h>

#include "parallel_execution_tbb.h"

namespace grppi{

/** 
\addtogroup filter_pattern
@{
*/

/**
\addtogroup filter_pattern_tbb TBB parallel filter pattern.
\brief TBB parallel implementation fo the \ref md_stream-filter pattern.
@{
*/

/**
\brief Invoke [stream filter keep pattern](@ref md_stream-filter pattern) on a data
sequence with sequential execution policy.
\tparam Generator Callable type for value generator.
\tparam Predicate Callable type for filter predicate.
\tparam Consumer Callable type for value consumer.
\param ex TBB parallel execution policy object.
\param generate_op Generator callable object.
\param predicate_op Predicate callable object.
\param consume_op Consumer callable object.
*/

template <typename Generator, typename Predicate, typename Consumer>
void keep(parallel_execution_tbb & ex, Generator generate_op,
          Predicate predicate_op, Consumer consume_op) 
{
  using namespace std;
  using generated_type = typename result_of<Generator()>::type;
  using item_type = pair<generated_type,long>;

  mpmc_queue<item_type> generated_queue{ex.queue_size,ex.lockfree};
  mpmc_queue<item_type> filtered_queue{ex.queue_size, ex.lockfree};

  //THREAD 1-(N-1) EXECUTE FILTER AND PUSH THE VALUE IF TRUE
  tbb::task_group filterers;
  for (int i=1; i< ex.num_threads-1; ++i) {
    filterers.run([&](){
      //dequeue a pair element - order
      auto item = generated_queue.pop();
      while (item.first) {
        if (predicate_op(item.first.value())) {
          filtered_queue.push(item);
        }
        else {
          filtered_queue.push({{}, item.second});
        }
        item = generated_queue.pop();
      }
      //If is the last element
      filtered_queue.push({item.first,-1});
    });
  }

  //LAST THREAD CALL FUNCTION OUT WITH THE FILTERED ELEMENTS
  filterers.run([&](){
    int done_tasks = 0;
    vector<item_type> item_buffer;
    long order = 0;
    auto item{filtered_queue.pop()};
    while (done_tasks!=ex.num_threads-1) {
      //If is an end of stream element
      if (!item.first && item.second==-1){
        done_tasks++;
        if (done_tasks==ex.num_threads-2) break;
      }
      else {
        //If the element is the next one to be procesed
        if (order==item.second) {
          if(item.first) {
            consume_op(*item.first);
          }
          order++;
        }
        else {
          //If the incoming element is disordered
          item_buffer.push_back(item);
        }
      }
      //Search in the vector for next elements
      // TODO: find+erase
      for (auto it=item_buffer.begin(); it<item_buffer.end(); ++it) {
        if(it->second==order) {
          if (it->first) {
            consume_op((*it).first.value());
          }
          item_buffer.erase(it);
          order++;
        }
      } 
      item = filtered_queue.pop();
    }
    while (item_buffer.size()>0) {
      // TODO: find+erase
      for (auto it=item_buffer.begin(); it<item_buffer.end(); ++it) {
        if(it->second==order) {
          if(it->first) {
            consume_op(it->first.value());
          }
          item_buffer.erase(it);
          order++;
        }
      }
    }
           
  });

  //THREAD 0 ENQUEUE ELEMENTS
  long order = 0;
  for (;;) {
    auto item = generate_op();
    generated_queue.push(make_pair(item,order));
    order++;
    if (!item) {
      for (int i=0; i<ex.num_threads-2; ++i) {
        generated_queue.push({item,-1});
      }
      break;
    }
  }
    
  filterers.wait();
}

/**
\brief Invoke [stream filter discard pattern](@ref md_stream-filter pattern) on a data
sequence with sequential execution policy.
\tparam Generator Callable type for value generator.
\tparam Predicate Callable type for filter predicate.
\tparam Consumer Callable type for value consumer.
\param ex TBB parallel execution policy object.
\param generate_op Generator callable object.
\param predicate_op Predicate callable object.
\param consume_op Consumer callable object.
*/

template <typename Generator, typename Predicate, typename Consumer>
void discard(parallel_execution_tbb & ex, Generator generate_op,
             Predicate predicate_op, Consumer consume_op) 
{
  keep(ex, 
    std::forward<Generator>(generate_op), 
    [&](auto val) { return !predicate_op(val); },
    std::forward<Consumer>(consume_op) 
  );
}

}

#endif

#endif
