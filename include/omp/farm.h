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

#ifndef GRPPI_OMP_FARM_H
#define GRPPI_OMP_FARM_H

#ifdef GRPPI_OMP
#include <experimental/optional>

#include "parallel_execution_omp.h"


namespace grppi
{

/**
\addtogroup farm_pattern
@{
*/

/**
\addtogroup farm_pattern_omp OpenMP parallel farm pattern
OpenMP implementation of the \ref md_farm.
@{
*/

/**
\brief Invoke [farm pattern](@ref md_farm) on a data stream with OpenMP parallel 
execution with a generator and a consumer.
\tparam Generator Callable type for the generation operation.
\tparam Consumer Callable type for the consume operation.
\param ex OpenMP arallel execution policy object.
\param generate_op Generator operation.
\param consume_op Consumer operation.
*/
template <typename Generator, typename Consumer>
void farm(parallel_execution_omp & ex, Generator generate_op, 
          Consumer consume_op) 
{
  using namespace std;
  using result_type = typename result_of<Generator()>::type;
  auto queue = ex.make_queue<result_type>();

  #pragma omp parallel
  {
    #pragma omp single nowait
    {
      for (int i = 0; i<ex.concurrency_degree(); i++) {
        #pragma omp task shared(queue)
        {
          auto item{queue.pop()};
          while (item) {
            consume_op(*item);
            item = queue.pop() ;
          }
        }
      }
		
      for (;;) {
        auto item = generate_op();
        queue.push(item) ;
        if (!item) {
          for (int i=1; i<ex.concurrency_degree(); ++i) {
            queue.push(item);
          }
          break;
        }
      }

      #pragma omp taskwait
    }
  }	
}

/**
\brief Invoke [farm pattern](@ref md_farm) on a data stream with OpenMP parallel 
execution with a generator, a transformer, and a consumer.
\tparam Generator Callable type for the generation operation.
\tparam Tranformer Callable type for the tranformation operation.
\tparam Consumer Callable type for the consume operation.
\param ex OpenMP Parallel execution policy object.
\param generate_op Generator operation.
\param transform_op Transformer operation.
\param consume_op Consumer operation.
*/
template <typename Generator, typename Transformer, typename Consumer>
void farm(parallel_execution_omp & ex, Generator generate_op, 
          Transformer transform_op , Consumer consume_op) 
{
  using namespace std;
  using namespace experimental;
  using result_type = typename result_of<Generator()>::type;
  using result_value_type = typename result_type::value_type;
  using transformed_value_type = 
      typename result_of<Transformer(result_value_type)>::type;
  using transformed_type = optional<transformed_value_type>;

  auto generated_queue = ex.make_queue<result_type>();
  auto transformed_queue = ex.make_queue<transformed_type>();
  atomic<int> done_threads{0};

  #pragma omp parallel
  {
    #pragma omp single nowait
    {
      for (int i=0; i<ex.concurrency_degree(); ++i) {
        #pragma omp task shared(generated_queue, transformed_queue, transform_op)
        {
          auto item{generated_queue.pop()};
          while (item) {
            transformed_queue.push(transformed_type{transform_op(*item)});
            item = generated_queue.pop( ) ;
          }
          generated_queue.push(item);
          done_threads++;
          if (done_threads == ex.concurrency_degree())
            transformed_queue.push(transformed_type{});
        }
      }

      #pragma omp task shared(transformed_queue,consume_op)
      { 
        auto item{transformed_queue.pop()};
        while (item) {
          consume_op( item.value() );
          item = transformed_queue.pop( );
        }
      }

      for (;;) {
        auto item = generate_op();
        generated_queue.push(item);
        if (!item) break;
      }

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
