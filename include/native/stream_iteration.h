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

#ifndef GRPPI_NATIVE_STREAM_ITERATION_H
#define GRPPI_NATIVE_STREAM_ITERATION_H

#include <thread>
#include <utility>
#include <memory>

#include "parallel_execution_native.h"

namespace grppi{ 

template<typename Generator, typename Predicate, typename Consumer, typename ...MoreTransformers>
void stream_iteration(parallel_execution_native &ex, Generator && generate_op, pipeline_info<parallel_execution_native , MoreTransformers...> && pipe, Predicate && predicate_op, Consumer && consume_op){
  using namespace std;
  using generated_type = typename std::result_of<Generator()>::type;
  mpmc_queue< generated_type > generated_queue{ex.queue_size,ex.lockfree};
  mpmc_queue< generated_type > transformed_queue{ex.queue_size,ex.lockfree};
  std::atomic<int> nelem {0};
  std::atomic<bool> send_finish {false};
  //Stream generator
  thread generator_task([&](){
    // Register the thread in the execution model
    ex.register_thread();
    for (;;) {
      auto item = generate_op();
      if (!item) {
        send_finish=true;
        break;
      }
      nelem++;
      generated_queue.push(item);
    }
    ex.deregister_thread();
  });

  vector<thread> pipe_threads;
  composed_pipeline< mpmc_queue<generated_type>, mpmc_queue<generated_type>, 0, MoreTransformers ...>
    (generated_queue, forward<pipeline_info<parallel_execution_native , MoreTransformers...> >(pipe) , transformed_queue, pipe_threads); 
 
  for (;;) {
    //If every element has been processed
    if (send_finish && nelem==0) {
      queue.push({});
      send_finish = false;
      break;
    }
    auto item{transformed_queue.pop()};
    //Check the predicate
    if (!predicate_op(*item)) {
      nelem--;
      consume_op(*item);
      //If the condition is not met reintroduce the element in the input queue
    }else queue.push(item);

  }
  generator_task.join();
  for(auto && t : pipe_threads){ 
    t.join();
  }
}

template<typename Generator, typename Transformer, typename Predicate, typename Consumer>
void stream_iteration(parallel_execution_native &ex, Generator && generate_op, farm_info<parallel_execution_native,Transformer> && farm, Predicate && predicate_op, Consumer && consume_op){
  using namespace std;
  using generated_type = typename std::result_of<Generator()>::type;
  mpmc_queue< generated_type > generated_queue{ex.queue_size,ex.lockfree};
  mpmc_queue< generated_type > transformed_queue{ex.queue_size,ex.lockfree};
  atomic<int> done_threads {0};
  vector<thread> tasks;
   //Stream generator
  thread generator_task([&](){
    // Register the thread in the execution model
    farm.exectype.register_thread();
    for (;;) {
      auto item = generate_op();
      generated_queue.push(item);
      if (!item) break;
    }
    //When generation is finished it starts working on the farm
    auto item{generated_queue.pop()};
    while (item) {
      auto out = *item;
      do {
        out = farm.task(out);
      } while (predicate_op(out));
      transformed_queue.push(out);
      item = generated_queue.pop();
    }
    done_threads++;
    if(done_treads == farm.exectype.num_threads) {
      transformed_queue.push({});
    }
    else {
      generated_queue.push(item);
    }
    // Deregister the thread in the execution model
    farm.exectype.deregister_thread();
  });
  //Farm workers
  for(int th = 1; th < farm.exectype.num_threads; th++) {
    tasks.emplace_back([&]() {
      // Register the thread in the execution model
      farm.exectype.register_thread();
      auto item{generated_queue.pop()};
      while (item) {
        auto out = *item;
        do {
          out = farm.task(out);
        } while (predicate_op(out));
        transformed_queue.push(out);
        item = generated_queue.pop();
      }
      nend++;
      if (nend == farm.exectype.num_threads) {
        transformed_queue.push({});
      }
      else {
        generated_queue.push(item);
      }
      // Deregister the thread in the execution model
      farm.exectype.deregister_thread();
    });
  }
  //Output function
  std::thread consumer_task([&](){
    for (;;){
     auto item = queue_out.pop();
     if(!item) break;
       consume_op(*item);
     }
  });
  //Join threads
  for(auto && t : tasks) t.join();
   generator_task.join();
   consumer_task.join();

}

template<typename Generator, typename Transformer, typename Predicate, typename Consumer>
void stream_iteration(parallel_execution_native &ex, Generator && generate_op, Transformer && transform_op, Predicate && predicate_op, Consumer && consume_op){
  using namespace std;
  using namespace std::experimental;
  using generated_type = typename result_of<Generator()>::type;
  using generated_value_type = typename generated_type::value_type;

  using transformed_type = typename result_of<Transformer(generated_value_type)>::type;
  
  mpmc_queue<generated_type> generated_queue{ex.queue_size,ex.lockfree};
  mpmc_queue<optional<transformed_type>> produced_queue{ex.queue_size,ex.lockfree};


  std::thread producer_task([&generate_op, &generated_queue](){
    for(;;) {
      auto item = generate_op();
      generated_queue.push(item);
      if (!item) break;
    }
  });
  
  std::thread transformer_task([&generated_queue,&transform_op,&predicate_op,&produced_queue](){
    auto item = generated_queue.pop();
    while (item) {
     auto val = *item;
     do {
       val = transform_op(val);
     } while (predicate_op(val));
     produced_queue.push(val);
     item = generated_queue.pop();
    }
    produced_queue.push({});
  });

  std::thread consumer_task([&consume_op,&produced_queue](){
    auto item = produced_queue.pop();
    while (item) {
      consume_op(*item);
      item=produced_queue.pop();
    }
  });  
  producer_task.join();
  transformer_task.join();
  consumer_task.join();

}

}
#endif
