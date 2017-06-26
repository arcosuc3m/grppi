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

#ifndef GRPPI_REDUCE_THR_H
#define GRPPI_REDUCE_THR_H

#include <thread>
#include <functional>

namespace grppi{

template < typename InputIt, typename ReduceOperator>
typename std::iterator_traits<InputIt>::value_type
reduce(parallel_execution_native &p, InputIt first, InputIt last, typename std::iterator_traits<InputIt>::value_type init, ReduceOperator && op){
   
    auto identityVal = init;

    int numElements = last - first;
    int elemperthr = numElements/p.num_threads;
    std::atomic<int> finishedTask(1);
    //local output
    std::vector<typename std::iterator_traits<InputIt>::value_type> out(p.num_threads);
    //Create threads
    for(int i=1;i<p.num_threads;i++){

      auto begin = first + (elemperthr * i);
      auto end = first + (elemperthr * (i+1));
      if(i == p.num_threads -1) end = last;
      p.pool.create_task(boost::bind<void>(
           [&](InputIt begin, InputIt end, int tid){
               out[tid] = identityVal;
               for( ; begin != end; begin++ ) {
                   out[tid] = op(out[tid], *begin );
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
         out[0] = op( out[0], *first);
    }

    while(finishedTask.load()!=p.num_threads);

    auto outVal = out[0];
    for(unsigned int i = 1; i < out.size(); i++){
       outVal = op(outVal, out[i]);
    }
    return outVal;
}

template < typename InputIt, typename ReduceOperator>
typename std::result_of< ReduceOperator(typename std::iterator_traits<InputIt>::value_type, typename std::iterator_traits<InputIt>::value_type) >::type
reduce(parallel_execution_native &p, InputIt first, InputIt last, ReduceOperator && op){
   auto identityVal = !op(false,true);
   return reduce(p, first, last, identityVal, std::forward<ReduceOperator>(op));
}



//typename ReduceOperator::result_type
/*template < typename InputIt, typename ReduceOperator>
typename std::result_of< ReduceOperator(typename std::iterator_traits<InputIt>::value_type, typename std::iterator_traits<InputIt>::value_type) >::type
reduce(parallel_execution_native &p, InputIt first, InputIt last, ReduceOperator op) {
    auto identityVal = !op(false,true);
    int numElements = last - first;
    int elemperthr = numElements/p.num_threads;
    std::atomic<int> finishedTask(1);
    //local output
    std::vector<typename ReduceOperator::result_type> out(p.num_threads);
    //Create threads
    for(int i=1;i<p.num_threads;i++){

      auto begin = first + (elemperthr * i);
      auto end = first + (elemperthr * (i+1));
      if(i == p.num_threads -1) end = last;
      p.pool.create_task(boost::bind<void>(
           [&](InputIt begin, InputIt end, int tid){
               out[tid] = identityVal;
               for( ; begin != end; begin++ ) {
                   out[tid] = op(out[tid], *begin );
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
         out[0] = op( out[0], *first);
    }

    while(finishedTask.load()!=p.num_threads);
    
    typename ReduceOperator::result_type outVal = out[0];
    for(unsigned int i = 1; i < out.size(); i++){
       outVal = op(outVal, out[i]);
    }
    return outVal;
}*/

}
#endif
