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

#ifndef GRPPI_NATIVE_REDUCE_H
#define GRPPI_NATIVE_REDUCE_H

#include "parallel_execution_native.h"

#include <thread>

namespace grppi {

/**
\addtogroup reduce_pattern
@{
\addtogroup reduce_pattern_native Native parallel reduce pattern
\brief Native parallel implementation of the \ref md_reduce.
@{
*/

/**
\brief Invoke \ref md_reduce with identity value
on a data sequence with parallel native execution.
\tparam InputIt Iterator type used for input sequence.
\tparam Identity Type for the identity value.
\tparam Combiner Callable type for the combiner operation.
\param ex Parallel native execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param identity Identity value for the combiner operation.
\param combiner_op Combiner operation for the reduction.
*/
template < typename InputIt, typename Identity, typename Combiner>
auto reduce(parallel_execution_native & ex,
            InputIt first, InputIt last, 
            Identity identity,
            Combiner && combine_op)
{
    auto identityVal = identity;

    int numElements = last - first;
    int elemperthr = numElements/ex.concurrency_degree();
    std::atomic<int> finishedTask(1);
    //local output
    std::vector<typename std::iterator_traits<InputIt>::value_type> out(ex.concurrency_degree());
    //Create threads
    for(int i=1;i<ex.concurrency_degree();i++){
      auto begin = first + (elemperthr * i);
      auto end = first + (elemperthr * (i+1));
      if(i == ex.concurrency_degree() -1) end = last;
      ex.pool.create_task(boost::bind<void>(
           [&](InputIt begin, InputIt end, int tid){
               out[tid] = identityVal;
               for( ; begin != end; begin++ ) {
                   out[tid] = combine_op(out[tid], *begin );
               }
               finishedTask++;
            },
            std::move(begin), std::move(end), i
      ));
    }
    //Main thread
    auto end = first + elemperthr;
    out[0] = identityVal;
    for(;first!=end;first++){
         out[0] = combine_op( out[0], *first);
    }

    while(finishedTask.load()!=ex.concurrency_degree());

    auto outVal = out[0];
    for(unsigned int i = 1; i < out.size(); i++){
       outVal = combine_op(outVal, out[i]);
    }
    return outVal;
}

/**
@}
@}
*/

}
#endif
