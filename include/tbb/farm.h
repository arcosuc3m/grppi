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

#ifndef GRPPI_TBB_FARM_H
#define GRPPI_TBB_FARM_H

#ifdef GRPPI_TBB

#include <experimental/optional>

#include <tbb/tbb.h>

#include "parallel_execution_tbb.h"

namespace grppi{

/**
\addtogroup farm_pattern
@{
*/

/**
\addtogroup farm_pattern_tbb TBB parallel farm pattern
\brief TBB parallel implementation of the \ref md_farm.
@{
*/

/**
\brief Invoke [farm pattern](@ref md_farm) on a data stream with TBB parallel 
execution with a generator and a consumer.
\tparam Generator Callable type for the generation operation.
\tparam Consumer Callable type for the consume operation.
\param ex TBB parallel execution policy object.
\param generate_op Generator operation.
\param consume_op Consumer operation.
*/
template <typename Generator, typename Consumer>
void farm(parallel_execution_tbb & ex, Generator generate_op, 
          Consumer consume_op) 
{
  using namespace std;

  using generated_type = typename result_of<Generator()>::type;
  auto queue = ex.make_queue<generated_type>();

  tbb::task_group g;
  for (int i=0; i<ex.concurrency_degree(); ++i) {
    g.run([&](){
      auto item{queue.pop()};
      while (item) {
        consume_op(*item);
        item = queue.pop();
      }
    });
  }

  //Generate elements
  for (;;) {
    auto item{generate_op()};
    queue.push(item);
      if (!item) {
        for (int i=1; i<ex.concurrency_degree(); i++) {
          queue.push(item);
        }
        break;
      }
    }

    //Join threads
    g.wait();
}

/**
\brief Invoke [farm pattern](@ref md_farm) on a data stream with TBB parallel 
execution with a generator and a consumer.
\tparam Generator Callable type for the generation operation.
\tparam Tranformer Callable type for the tranformation operation.
\tparam Consumer Callable type for the consume operation.
\param ex TBB parallel execution policy object.
\param generate_op Generator operation.
\param transform_op Transformer operation.
\param consume_op Consumer operation.
*/
template <typename Generator, typename Transformer, typename Consumer>
void farm(parallel_execution_tbb & ex, Generator generate_op, 
          Transformer transform_op, Consumer consume_op) 
{
  using namespace std;
  using namespace experimental;
  using generated_type = typename result_of<Generator()>::type;
  using generated_value_type = typename generated_type::value_type;
  using transformed_value_type = 
      typename result_of<Transformer(generated_value_type)>::type;
  using transformed_type = optional<transformed_value_type>;

  auto generated_queue = ex.make_queue<generated_type>();
  auto transformed_queue = ex.make_queue<transformed_type>();

  atomic<int>done_threads{0};
  tbb::task_group generators;
  for (int i=0; i<ex.concurrency_degree(); ++i) {
    generators.run([&](){
      auto item{generated_queue.pop()};
      while (item) {
        auto result = transform_op(*item);
        transformed_queue.push(transformed_type{result});
        item = generated_queue.pop();
      }
      done_threads++;
      if (done_threads==ex.concurrency_degree()) {
        transformed_queue.push(transformed_type{});
      }
    });
  }

  thread consumer_thread([&](){
    auto item {transformed_queue.pop()};
    while (item) {
      consume_op(*item);
      item = transformed_queue.pop(  );
    }
  });

   //Generate elements
  for (;;) {
    auto item = generate_op();
    generated_queue.push(item) ;
    if(!item) {
      for (int i=1; i<ex.concurrency_degree(); ++i) {
        generated_queue.push(item) ;
      }
      break;
    }
  }

  generators.wait();
  consumer_thread.join();
}

/**
@}
@}
*/

}
#endif

#endif
