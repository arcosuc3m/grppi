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

#include "parallel_execution_native.h"

#include <thread>
#include <utility>
#include <memory>

namespace grppi { 

/**
\addtogroup stream_iteration_pattern
@{
\addtogroup stream_iteration_pattern_native Native parallel stream iteration pattern
\brief Sequential implementation of the \ref md_stream-iteration.
@{
*/

/**
\brief Invoke \ref md_stream-iteration on a data stream with native parallel 
execution with a generator, a predicate, a consumer and a pipeline as a transformer.
\tparam Generator Callable type for the generation operation.
\tparam Predicate Callable type for the predicate operation.
\tparam Consumer Callable type for the consume operation.
\tparam MoreTransformers Callable type for the transformer operations.
\param ex Parallel native execution policy object.
\param generate_op Generator operation.
\param predicate_op Predicate operation.
\param consume_op Consumer operation.
\param pipe Composed pipeline object.
*/
template<typename Generator, typename Predicate, typename Consumer, 
         typename ...MoreTransformers>
void repeat_until(parallel_execution_native & ex, 
                  Generator && generate_op, 
                  pipeline_info<parallel_execution_native, 
                      MoreTransformers...> && pipe, 
                  Predicate predicate_op, Consumer consume_op)
{
  using namespace std;
  using generated_type = typename result_of<Generator()>::type;
  using pipeline_info_type = pipeline_info<parallel_execution_native, MoreTransformers...>;

  auto generated_queue = ex.make_queue<generated_type>();
  auto transformed_queue = ex.make_queue<generated_type>();
  atomic<int> num_elements{0};
  atomic<bool> send_finish{false};

  thread generator_task([&](){
    auto manager = ex.thread_manager();
    for (;;) {
      auto item{generate_op()};
      if (!item) {
        send_finish=true;
        break;
      }
      num_elements++;
      generated_queue.push(item);
    }
  });

  vector<thread> pipe_threads;
  composed_pipeline<decltype(generated_queue), 
          decltype(transformed_queue), 0, MoreTransformers ...>(
      generated_queue, 
      forward<pipeline_info_type>(pipe), 
      transformed_queue, pipe_threads); 

  auto manager = ex.thread_manager();
  for (;;) {
    //If every element has been processed
    if (send_finish && num_elements==0) {
      generated_queue.push({});
      send_finish = false;
      break;
    }

    auto item{transformed_queue.pop()};
    if (predicate_op(*item)) {
      num_elements--;
      consume_op(*item);
    }
    else {
      //If the condition is not met reintroduce the element in the input queue
      generated_queue.push(item);
    }
  }

  generator_task.join();
  for (auto && t : pipe_threads) { t.join(); }
}

/**
\brief Invoke \ref md_stream-iteration on a data stream with native parallel 
execution with a generator, a predicate, a consumer and a farm as a transformer.
\tparam Generator Callable type for the generation operation.
\tparam Predicate Callable type for the predicate operation.
\tparam Consumer Callable type for the consume operation.
\tparam Transformer Callable type for the transformer operations.
\param ex Parallel native execution policy object.
\param generate_op Generator operation.
\param predicate_op Predicate operation.
\param consume_op Consumer operation.
\param farm Composed farm object.
*/
template<typename Generator, typename Transformer, typename Predicate, 
         typename Consumer>
void repeat_until(parallel_execution_native &ex, 
                  Generator generate_op, 
                  farm_info<parallel_execution_native,Transformer> && farm, 
                  Predicate predicate_op, Consumer consume_op)
{
  using namespace std;
  using generated_type = typename result_of<Generator()>::type;
  auto generated_queue = ex.make_queue<generated_type>();
  auto transformed_queue = ex.make_queue<generated_type>();
  atomic<int> done_threads{0};
  vector<thread> tasks;

  thread generator_task([&](){
    auto manager = farm.exectype.thread_manager();
    for (;;) {
      auto item = generate_op();
      generated_queue.push(item);
      if (!item) break;
    }

    //When generation is finished start working on the farm
    auto item{generated_queue.pop()};
    while (item) {
      auto out = *item;
      do {
        out = farm.task(out);
      } while (!predicate_op(out));
      transformed_queue.push(out);
      item = generated_queue.pop();
    }
    done_threads++;
    if(done_threads == farm.exectype.concurrency_degree()) {
      transformed_queue.push({});
    }
    else {
      generated_queue.push(item);
    }
  });
  //Farm workers
  for(int th = 1; th < farm.exectype.concurrency_degree(); th++) {
    tasks.emplace_back([&]() {
      auto manager = farm.exectype.thread_manager();
      auto item{generated_queue.pop()};
      while (item) {
        auto out = *item;
        do {
          out = farm.task(out);
        } while (!predicate_op(out));
        transformed_queue.push(out);
        item = generated_queue.pop();
      }
      done_threads++;
      if (done_threads == farm.exectype.concurrency_degree()) {
        transformed_queue.push({});
      }
      else {
        generated_queue.push(item);
      }
    });
  }

  thread consumer_task([&](){
    auto manager = ex.thread_manager();
    for (;;){
     auto item{transformed_queue.pop()};
     if(!item) break;
       consume_op(*item);
     }
  });

  for(auto && t : tasks) { t.join(); }
  generator_task.join();
  consumer_task.join();
}


/**
\brief Invoke \ref md_stream-iteration on a data stream with native parallel 
execution with a generator, a predicate, a transformer and a consumer.
\tparam Generator Callable type for the generation operation.
\tparam Predicate Callable type for the predicate operation.
\tparam Consumer Callable type for the consume operation.
\tparam Transformer Callable type for the transformer operation.
\param ex Parallel native execution policy object.
\param generate_op Generator operation.
\param predicate_op Predicate operation.
\param consume_op Consumer operation.
\param tranformer_op Tranformer operation.
*/
template<typename Generator, typename Transformer, typename Predicate, 
         typename Consumer>
void repeat_until(parallel_execution_native &ex, 
                  Generator generate_op, Transformer transform_op, 
                  Predicate predicate_op, Consumer consume_op) 
{
  using namespace std;
  using namespace experimental;
  using generated_type = typename result_of<Generator()>::type;
  using generated_value_type = typename generated_type::value_type;
  using transformed_type = 
      typename result_of<Transformer(generated_value_type)>::type;
  
  auto generated_queue = ex.make_queue<generated_type>();
  auto transformed_queue = ex.make_queue<generated_type>();

  thread producer_task([&generate_op, &generated_queue, &ex](){
    auto manager = ex.thread_manager();
    for(;;) {
      auto item{generate_op()};
      generated_queue.push(item);
      if (!item) break;
    }
  });
  
  thread transformer_task([&generated_queue,&transform_op,&predicate_op,
          &transformed_queue, &ex](){
    auto manager = ex.thread_manager();
    auto item{generated_queue.pop()};
    while (item) {
     auto val = *item;
     do {
       val = transform_op(val);
     } while (!predicate_op(val));
     transformed_queue.push(val);
     item = generated_queue.pop();
    }
    transformed_queue.push({});
  });

  thread consumer_task([&consume_op,&transformed_queue,&ex](){
    auto manager = ex.thread_manager();
    auto item{transformed_queue.pop()};
    while (item) {
      consume_op(*item);
      item=transformed_queue.pop();
    }
  });  

  producer_task.join();
  transformer_task.join();
  consumer_task.join();
}

}

#endif
