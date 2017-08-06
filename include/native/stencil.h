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

#ifndef GRPPI_NATIVE_STENCIL_H
#define GRPPI_NATIVE_STENCIL_H

#include "parallel_execution_native.h"
#include "../common/iterator.h"

namespace grppi {

/**
\addtogroup stencil_pattern
@{
\addtogroup stencil_pattern_native Native parallel stencil pattern
\brief Native parallel implementation of the \ref md_stencil.
@{
*/

/**
\brief Invoke \ref md_stencil on a data sequence with 
native parallel execution.
\tparam InputIt Iterator type used for the input sequence.
\tparam OutputIt Iterator type used for the output sequence
\tparam Neighbourhood Callable type for obtaining the neighbourhood.
\tparam StencilTransformer Callable type for performing the stencil transformation.
\param ex Native parallel execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param out Iterator to the first element in the output sequence.
\param transform_op Stencil transformation operation.
\param neighbour_op Neighbourhood operation.
*/
template <typename InputIt, typename OutputIt, typename StencilTransformer, 
          typename Neighbourhood>
void stencil(parallel_execution_native & ex, 
             InputIt first, InputIt last, OutputIt first_out, 
             StencilTransformer transform_op, Neighbourhood neighbour_op) 
{
  using namespace std;

  vector<thread> tasks;
  int size = distance(first,last);
  int elements_per_thread = size/ex.concurrency_degree();
 
  for (int i=1; i<ex.concurrency_degree(); ++i) {
    auto begin = first + (elements_per_thread * i);
    auto end = (i==ex.concurrency_degree()-1)?
        last :
        next(first, elements_per_thread * (i+1));

    auto out = first_out + (elements_per_thread * i);

    tasks.emplace_back([&](InputIt begin, InputIt end, OutputIt out) {
      auto manager = ex.thread_manager();

      while (begin!=end) {
        *out = transform_op(begin, neighbour_op(begin));
        begin++;
        out++;
      }
    }, 
    begin, end, out);
  }

  auto end = first + elements_per_thread;
  while(first!=end){
    *first_out = transform_op(first, neighbour_op(first));
    first++;
    first_out++;
  }

  for (auto && t : tasks) { t.join(); }
}

/**
\brief Invoke \ref md_stencil on multiple data sequences with 
native parallel execution.
\tparam InputIt Iterator type used for the input sequence.
\tparam OutputIt Iterator type used for the output sequence
\tparam Neighbourhood Callable type for obtaining the neighbourhood.
\tparam StencilTransformer Callable type for performing the stencil transformation.
\tparam OtherInputIts Iterator types for additional input sequences.
\param ex Native parallel execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param out Iterator to the first element in the output sequence.
\param transform_op Stencil transformation operation.
\param neighbour_op Neighbourhood operation.
\param other_firsts Iterators to the first element of additional input sequences.
*/
template <typename InputIt, typename OutputIt, typename StencilTransformer, 
          typename Neighbourhood, typename ... OtherInputIts>
void stencil(parallel_execution_native & ex, 
             InputIt first, InputIt last, OutputIt first_out, 
             StencilTransformer transform_op, Neighbourhood neighbour_op, 
             OtherInputIts ... other_firsts ) 
{
  using namespace std;

  vector<thread> tasks;
  int size = last - first;
  int elements_per_thread = size/ex.concurrency_degree();

  for (int i=1; i<ex.concurrency_degree(); i++){
  auto begin = first + (elements_per_thread * i);
  auto end = (i==ex.concurrency_degree()-1)?
      last :
      first + elements_per_thread * (i+1);

  auto out = first_out + (elements_per_thread * i);
        
  tasks.emplace_back(
    [&](InputIt begin, InputIt end, OutputIt out, int i, int n, OtherInputIts ... other_firsts){
      auto manager = ex.thread_manager();

      advance_iterators(n*i, other_firsts ...);
      while (begin!=end) {
        *out = transform_op(begin, neighbour_op(begin,other_firsts...));
        begin++;
        advance_iterators(other_firsts ... );
        out++;
      }
    },
    begin, end, out, i, elements_per_thread,other_firsts ...);
  }

  auto end = first + elements_per_thread;
  while (first!=end) {
    *first_out = transform_op(first, neighbour_op(first,other_firsts...));
    first++;
    advance_iterators( other_firsts ... );
    first_out++;
  }


  for (auto && t : tasks) { t.join(); }
}

/**
@}
@}
*/

}
#endif
