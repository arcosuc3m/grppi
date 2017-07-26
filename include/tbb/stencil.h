/**
* @version		GrPPI v0.2
* @copyright		Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license		GNU/GPL, see LICENSE.txt
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your transform_option) any later version.
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

#ifndef GRPPI_TBB_STENCIL_H
#define GRPPI_TBB_STENCIL_H

#ifdef GRPPI_TBB

#include <tbb/tbb.h>

#include "parallel_execution_tbb.h"

namespace grppi{

template <typename InputIt, typename OutputIt, typename StencilTransformer, 
          typename Neighbourhood, typename ... OtherInputIts>
void stencil_impl(parallel_execution_tbb & ex, 
                  int elements_per_thread, int index, 
                  InputIt first, InputIt last, OutputIt first_out, 
                  StencilTransformer transform_op, Neighbourhood neighbour_op, 
                  OtherInputIts ... other_firsts){
  auto begin = next(first, elements_per_thread * index);
  auto end = (index==ex.concurrency_degree()-1)?
      last :
      next(first, elements_per_thread * (index+1));
  auto out = next(first_out, elements_per_thread * index);
  advance_iterators(elements_per_thread* index, other_firsts ...);
  while(begin!=end){
    *out = transform_op(begin, neighbour_op(begin, other_firsts ...));
    begin++;
    advance_iterators(other_firsts...);
    out++;
  }
}


/**
\addtogroup stencil_pattern
@{
*/

/**
\addtogroup stencil_pattern_tbb TBB Parallel stencil pattern
\brief TBB parallel implementation of the \ref md_stencil pattern.
@{
*/

/**
\brief Invoke [stencil pattern](\ref md_stencil) on a data sequence with 
TBB parallel execution.
\tparam InputIt Iterator type used for the input sequence.
\tparam OutputIt Iterator type used for the output sequence
\tparam Neighbourhood Callable type for obtaining the neighbourhood.
\tparam StencilTransformer Callable type for performing the stencil transformation.
\param ex TBB parallel execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param out Iterator to the first element in the output sequence.
\param transform_op Stencil transformation transform_operation.
\param neighbour_op Neighbourhood transform_operation.
*/
template <typename InputIt, typename OutputIt, typename StencilTransformer, 
          typename Neighbourhood>
void stencil(parallel_execution_tbb & ex, 
             InputIt first, InputIt last, OutputIt first_out, 
             StencilTransformer transform_op, 
             Neighbourhood neighbour_op) 
{
  int size = last - first;
  int elements_per_thread = size/ex.concurrency_degree();
  tbb::task_group g;

  for (int i=1; i<ex.concurrency_degree(); ++i) {
    g.run(
      [&neighbour_op, &transform_op, first, first_out, elements_per_thread, 
          i, last, ex]() {
        auto begin = first + (elements_per_thread * i);
        auto end = (i==ex.concurrency_degree()-1)?
            last :
            next(first, elements_per_thread * (i+1));

        auto out = next(first_out, elements_per_thread * i);
        while (begin!=end) {
          *out = transform_op(begin, neighbour_op(begin));
          begin++;
          out++;
        }
      }
    );
  }

  auto end = first + elements_per_thread;
  while (first!=end) {
    *first_out = transform_op(first, neighbour_op(first));
    first++;
    first_out++;
  }

  g.wait();
}

/**
\brief Invoke [stencil pattern](\ref md_stencil) on multiple data sequences with 
TBB parallel execution.
\tparam InputIt Iterator type used for the input sequence.
\tparam OutputIt Iterator type used for the output sequence
\tparam Neighbourhood Callable type for obtaining the neighbourhood.
\tparam StencilTransformer Callable type for performing the stencil transformation.
\tparam OtherInputIts Iterator types for additional input sequences.
\param ex TBB parallel execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param out Iterator to the first element in the output sequence.
\param transform_op Stencil transformation operation.
\param neighbour_op Neighbourhood operation.
\param other_firsts Iterators to the first element of additional input sequences.
*/
template <typename InputIt, typename OutputIt, typename StencilTransformer, 
          typename Neighbourhood, typename ... OtherInputIts>
void stencil(parallel_execution_tbb & ex, 
             InputIt first, InputIt last, OutputIt first_out, 
             StencilTransformer transform_op, Neighbourhood neighbour_op, 
             OtherInputIts ... other_firsts ) 
{
  int size = distance(first,last);
  int elements_per_thread = size/ex.concurrency_degree();
  tbb::task_group g;
  for(int index=1; index<ex.concurrency_degree(); ++index) {
    g.run(
      [neighbour_op, transform_op, first, first_out, elements_per_thread, 
          index, last, &ex, other_firsts...]()
      {
        stencil_impl(ex, elements_per_thread, index,
            first, last, first_out, transform_op, 
            neighbour_op, other_firsts...);
      }
    );
  }

  auto end = next(first, elements_per_thread);
  while(first!=end){
    *first_out = transform_op(first, neighbour_op(first,other_firsts ...));
    first++;
    advance_iterators( other_firsts ... );
    first_out++;
  }

  g.wait();
}

}

#endif

#endif
