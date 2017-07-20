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

#ifndef GRPPI_NATIVE_FARM_H
#define GRPPI_NATIVE_FARM_H

#include <experimental/optional>

#include <thread>
#include <utility>
#include <memory>

#include "parallel_execution_native.h"

namespace grppi{

/**
\addtogroup farm_pattern
@{
*/

/**
\addtogroup farm_pattern_native Native parallel farm pattern
Sequential implementation of the \ref md_farm.
@{
*/

/**
\brief Invoke [farm pattern](@ref md_farm) on a data stream with native parallel 
execution with a generator and a consumer.
\tparam Generator Callable type for the generation operation.
\tparam Consumer Callable type for the consume operation.
\param ex Parallel native execution policy object.
\param generate_op Generator operation.
\param consume_op Consumer operation.
*/
template <typename Generator, typename Consumer>
void farm(parallel_execution_native & ex, Generator generate_op, 
          Consumer consume_op) 
{
  using namespace std;
  using result_type = typename result_of<Generator()>::type;
  mpmc_queue<result_type> queue{ex.queue_size,ex.lockfree};

  vector<thread> tasks;
  for (int i=0; i<ex.num_threads; ++i) {
    tasks.emplace_back([&](){
      ex.register_thread();

      auto item{queue.pop()};
      while(item) {
        consume_op(*item);
        item = queue.pop();
      }
      queue.push(item);

      ex.deregister_thread();
    });
  }

  for (;;) {
    auto item{generate_op()};
    queue.push(item);
    if (!item) break;
  }

  for (auto && t : tasks) { t.join(); }
}

/**
\brief Invoke [farm pattern](@ref md_farm) on a data stream with native parallel 
execution with a generator and a consumer.
\tparam Generator Callable type for the generation operation.
\tparam Tranformer Callable type for the tranformation operation.
\tparam Consumer Callable type for the consume operation.
\param ex Parallel native execution policy object.
\param generate_op Generator operation.
\param transform_op Transformer operation.
\param consume_op Consumer operation.
*/
template <typename Generator, typename Transformer, typename Consumer>
void farm(parallel_execution_native & ex, Generator generate_op, 
          Transformer transform_op , Consumer consume_op) 
{
  using namespace std;
  using namespace experimental;
  using generated_type = typename result_of<Generator()>::type;
  using generated_value_type = typename generated_type::value_type;
  using transformed_value_type = 
      typename result_of<Transformer(generated_value_type)>::type;
  using transformed_type = optional<transformed_value_type>;

  mpmc_queue<generated_type> generated_queue{ex.queue_size,ex.lockfree};
  mpmc_queue<transformed_type> transformed_queue{ex.queue_size, ex.lockfree};

  atomic<int> done_threads(0);
  vector<thread> tasks;

  for (int i=0; i<ex.num_threads; ++i) {
    tasks.emplace_back([&](){
      ex.register_thread();

      auto item{generated_queue.pop()};
      while (item) {
        transformed_queue.push(transformed_type{transform_op(*item)});
        item = generated_queue.pop();
      }
      generated_queue.push(item);
      done_threads++;
      if (done_threads==ex.num_threads) {
        transformed_queue.push(transformed_type{});
      }

      ex.deregister_thread();
    });
  }

  tasks.emplace_back([&](){
    ex.register_thread();

    auto item{transformed_queue.pop()};
    while (item) {
      consume_op( item.value() );
      item = transformed_queue.pop( );
    }

    ex.deregister_thread();
  });

  for (;;) {
    auto  item{generate_op()};
    generated_queue.push(item);
    if(!item) break;
  }

  for (auto && t : tasks) { t.join(); }
}

}

#endif
