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

#ifndef GRPPI_NATIVE_MAP_H
#define GRPPI_NATIVE_MAP_H

#include "parallel_execution_native.h"

namespace grppi{

/**
\addtogroup map_pattern
@{
\addtogroup map_pattern_native Native parallel map pattern.
Implementation of map pattern for native parallel back-end.
@{
*/

/**
\brief Invoke [map pattern](@ref map-pattern) on a data sequence with native
paralell execution.
\tparam InputIt Iterator type used for input sequence.
\tparam OtuputIt Iterator type used for the output sequence.
\tparam Transformer Callable type for the transformation operation.
\param ex Native parallel execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param first_out Iterator to first elemento of the output sequence.
\param transf_op Transformation operation.
*/
template <typename InputIt, typename OutputIt, typename Transformer>
void map(parallel_execution_native & ex, 
         InputIt first, InputIt last, OutputIt first_out, 
         Transformer && transf_op)
{
  std::vector<std::thread> tasks;
  int numElements = last - first; 
  int elemperthr = numElements / ex.concurrency_degree(); 

  for(int i=1;i<ex.concurrency_degree();i++){
    auto begin = first + (elemperthr * i); 
    auto end = first + (elemperthr * (i+1)); 

    if(i == ex.concurrency_degree()-1 ) end= last;

    auto out = first_out + (elemperthr * i);
    tasks.emplace_back([&](InputIt begin, InputIt end, OutputIt out) {
      auto manager = ex.thread_manager();
          
      while (begin!=end) {
        *out = transf_op(*begin);
        begin++;
        out++;
      }
    }, begin, end, out);
  }
  //Map main threads
  auto end = first+elemperthr;
  while(first!=end) {
    *first_out = transf_op(*first);
    first++;
    first_out++;
  }

  //Join threads
  for(int i=0;i<ex.concurrency_degree()-1;i++){
    tasks[i].join();
  }
}

/**
\brief Invoke [map pattern](@ref map-pattern) on a data sequence with native
parallel execution.
\tparam InputIt Iterator type used for input sequence.
\tparam OtuputIt Iterator type used for the output sequence.
\tparam Transformer Callable type for the transformation operation.
\tparam OtherInputIts Iterator types used for additional input sequences.
\param ex Native parallel execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param first_out Iterator to first elemento of the output sequence.
\param transf_op Transformation operation.
\param more_firsts Additional iterators with first elements of additional sequences.
*/
template <typename InputIt, typename OutputIt, typename Transformer,
          typename ... OtherInputIts> 
void map(parallel_execution_native& ex, 
         InputIt first, InputIt last, OutputIt first_out, 
         Transformer && transf_op, 
         OtherInputIts ... more_inputs)
{
  std::vector<std::thread> tasks;

  //Calculate number of elements per thread
  int numElements = last - first;
  int elemperthr = numElements / ex.concurrency_degree();

  //Create tasks
  for(int i=1;i<ex.concurrency_degree();i++){
    //Calculate local input and output iterator 
    auto begin = first + (elemperthr * i);
    auto end = first + (elemperthr * (i+1));
    if( i == ex.concurrency_degree()-1) end = last;
    auto out = first_out + (elemperthr * i);
    //Begin task
    tasks.emplace_back([&](InputIt begin, InputIt end, OutputIt out, 
      int tid, int nelem, OtherInputIts ... more_inputs) {
        auto manager = ex.thread_manager();
      advance_iterators(nelem*tid, more_inputs ...);
      while (begin!=end) {
        *out = transf_op(*begin, *more_inputs ...);
        advance_iterators(more_inputs ...);
        begin++;
        out++;
      }
    }, begin, end, out, i, elemperthr, more_inputs...);
    //End task
  }

  //Map main thread
  auto end = first + elemperthr;
  while(first!=end) {
    *first_out = transf_op(*first, *more_inputs ...);
    advance_iterators(more_inputs ...);
    first++;
    first_out++;
  }

  //Join threads
  for(int i=0;i<ex.concurrency_degree()-1;i++) {
    tasks[i].join();
  }
}

/**
@}
@}
*/
}

#endif
