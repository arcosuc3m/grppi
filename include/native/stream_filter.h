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

#ifndef GRPPI_NATIVE_STREAM_FILTER_H
#define GRPPI_NATIVE_STREAM_FILTER_H

#include "parallel_execution_native.h"

namespace grppi{

/** 
\addtogroup filter_pattern
@{
*/

/**
\addtogroup filter_pattern_native Native parallel filter pattern.
\brief Native parallel implementation fo the \ref md_stream-filter pattern.
@{
*/

/**
\brief Invoke [stream filter pattern](@ref md_stream-filter pattern) on a data
sequence with sequential execution policy.
\tparam Generator Callable type for value generator.
\tparam Predicate Callable type for filter predicate.
\tparam Consumer Callable type for value consumer.
\param ex Native parallel execution policy object.
\param generate_op Generator callable object.
\param predicate_op Predicate callable object.
\param consume_op Consumer callable object.
*/
template <typename Generator, typename Predicate, typename Consumer>
void stream_filter(parallel_execution_native & ex, Generator generate_op, 
                   Predicate predicate_op, Consumer consume_op) 
{
  using namespace std;
  using generated_type = typename result_of<Generator()>::type;
  using item_type = pair<generated_type,long>;

  mpmc_queue<item_type> generated_queue{ex.queue_size,ex.lockfree};
  mpmc_queue<item_type> filtered_queue{ex.queue_size,ex.lockfree};

  //THREAD 1-(N-1) EXECUTE FILTER AND PUSH THE VALUE IF TRUE
  vector<thread> tasks;
  for (int i=0; i<ex.concurrency_degree()-1; ++i) {
    tasks.emplace_back([&](){
      ex.register_thread();

      // queue a pair element - order
      auto item{generated_queue.pop()};

      while (item.first) {
        if(predicate_op(*item.first)) {
          filtered_queue.push(item);
        }
        else {
          filtered_queue.push(make_pair(generated_type{}, item.second));
        }
        item = generated_queue.pop();
      }
      //If is the last element
      filtered_queue.push(make_pair(item.first, -1 ));

      ex.deregister_thread();
    });
  }

  //LAST THREAD CALL FUNCTION OUT WITH THE FILTERED ELEMENTS
  thread consumer([&](){
    ex.register_thread();

    int done_threads = 0; 
    
    vector<item_type> item_buffer;
    long order = 0;

    // queue an element
    auto item{filtered_queue.pop()};
    while (done_threads != ex.concurrency_degree()-1) {
      //If is an end of stream element
      if (!item.first && item.second==-1) {
        done_threads++;
        if (done_threads==ex.concurrency_degree()-1) break;
      }
      //If there is not an end element
      else {
        //If the element is the next one to be procesed
        if (order==item.second) {
          if (item.first) {
            consume_op(*item.first);
          }
          order++;
        }
        else {
          //If the incoming element is out of order
          item_buffer.push_back(item);
        }
      }

      //Search in the buffer for next elements
      auto itrm = remove_if(begin(item_buffer), end(item_buffer),
        [&order](auto & item) {
          bool res = item.second == order;
          if (res) order++;
          return res;
        }
      );
      for_each (itrm, end(item_buffer), 
        [&consume_op](auto & item) {
          if (item.first) { consume_op(*item.first); }
        }
      );
      item_buffer.erase(itrm, end(item_buffer));

      item = filtered_queue.pop();
    }

    for (;;) {
      auto it_find = find_if(begin(item_buffer), end(item_buffer),
          [order](auto & item) { return item.second == order; });
      if (it_find == end(item_buffer)) break;
      if (it_find->first) {
        consume_op(*it_find->first);
      }
      item_buffer.erase(it_find);
      order++;
    }
           
    ex.deregister_thread();
  });

  //THREAD 0 ENQUEUE ELEMENTS
  long order = 0;
  for (;;) {
    auto item = generate_op();
    generated_queue.push(make_pair(item,order));
    order++;
    if(!item) {
      for (int i=0; i<ex.concurrency_degree()-1; ++i) {
        generated_queue.push(make_pair(item,-1));
      }
      break;
    }
  }

  for (auto && t : tasks) { t.join(); }
  consumer.join();
}

}
#endif
