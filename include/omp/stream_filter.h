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

#ifndef GRPPI_OMP_STREAM_FILTER_H
#define GRPPI_OMP_STREAM_FILTER_H

#ifdef GRPPI_OMP

#include "parallel_execution_omp.h"

namespace grppi{

/** 
\addtogroup filter_pattern
@{
*/

/**
\addtogroup filter_pattern_omp OpenMP parallel filter pattern.
\brief OpenMP parallel implementation fo the \ref md_stream-filter pattern.
@{
*/

/**
\brief Invoke [stream filter pattern](@ref md_stream-filter pattern) on a data
sequence with sequential execution policy.
\tparam Generator Callable type for value generator.
\tparam Predicate Callable type for filter predicate.
\tparam Consumer Callable type for value consumer.
\param ex OpenMP parallel execution policy object.
\param generate_op Generator callable object.
\param predicate_op Predicate callable object.
\param consume_op Consumer callable object.
*/
template <typename Generator, typename Predicate, typename Consumer>
void stream_filter(parallel_execution_omp & ex, Generator generate_op, 
                   Predicate predicate_op, Consumer consume_op) 
{
  using namespace std;
  using generated_type = typename result_of<Generator()>::type;

  mpmc_queue<generated_type> generated_queue{ex.queue_size, ex.lockfree};
  mpmc_queue<generated_type> filtered_queue{ex.queue_size, ex.lockfree};

  #pragma omp parallel
  {
    #pragma omp single nowait 
    {
      //THREAD 1-(N-1) EXECUTE FILTER AND PUSH THE VALUE IF TRUE
      for (int i=0; i< ex.num_threads - 1; i++) {
        #pragma omp task shared(generated_queue, filtered_queue)
        {
          auto item{generated_queue.pop()};
          while (item) {
            if (predicate_op(item.value())) {
              filtered_queue.push(item);
            }
            item = generated_queue.pop();
          }
          filtered_queue.push(item);
        }
      }

      #pragma omp task shared(filtered_queue)
      {
        //LAST THREAD CALL FUNCTION OUT WITH THE FILTERED ELEMENTS
        int done_threads = 0;
        auto item{filtered_queue.pop()};
        while (done_threads!=ex.num_threads-1) {
          if (!item) {
            done_threads++;
            if(done_threads == ex.num_threads - 1) break;
          }
          else {
            consume_op(item.value());
          }
          item = filtered_queue.pop();
        }
      }

      //THREAD 0 ENQUEUE ELEMENTS
      for (;;) {
        auto item{generate_op()};
        generated_queue.push(item);
        if (!item) {
          for (int i = 0; i< ex.num_threads-1; i++) {
            generated_queue.push(item);
          }
          break;
        }
      }

      #pragma omp taskwait
    }
  }
}

}

#endif

#endif
