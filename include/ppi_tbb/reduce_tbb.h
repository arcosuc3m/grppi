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

#ifndef GRPPI_REDUCE_TBB_H
#define GRPPI_REDUCE_TBB_H

#ifdef GRPPI_TBB

#include <tbb/tbb.h>

using namespace std;

//typename std::enable_if<!is_iterator<Output>::value, bool>::type,
namespace grppi{
template < typename InputIt, typename Output,typename ReduceOperator>
 typename std::enable_if<!is_iterator<Output>::value, void>::type 
reduce(parallel_execution_tbb const &p, InputIt first, InputIt last, Output & firstOut, ReduceOperator op) {
   typename ReduceOperator::result_type identityVal = !op(false,true); 
   //FIXME: Necesita el valor inicial de la operacion
   firstOut = op(firstOut, 
                 tbb::parallel_reduce(tbb::blocked_range<InputIt>( first, last ), identityVal, 
                 [&](const tbb::blocked_range<InputIt> &r, Output temp){   
                   for(InputIt i=r.begin(); i!= r.end(); ++i){
                       temp = op(temp,*i);
                   }
                   return temp;
                 },
                 [&](Output a, Output b) -> Output {
                   a = op(a,b);
                   return a;
                 }
                )
              );
}



template < typename InputIt, typename ReduceOperator>
 typename ReduceOperator::result_type
reduce(parallel_execution_tbb const &p, InputIt first, InputIt last, ReduceOperator op) {
   typename ReduceOperator::result_type identityVal = !op(false, true); 
   //FIXME: Necesita el valor inicial de la operacion
   return tbb::parallel_reduce(tbb::blocked_range<InputIt>( first, last ), identityVal,
              [&](const tbb::blocked_range<InputIt> &r, typename ReduceOperator::result_type  temp){
                 for(InputIt i=r.begin(); i!= r.end(); ++i){
                   temp = op( temp, *i);
                 }
                 return temp;
              },
              [&](typename ReduceOperator::result_type a, typename ReduceOperator::result_type b) -> typename ReduceOperator::result_type
              {
                a = op(a,b);
                return a;
              }
          );
   
}


template < typename InputIt, typename OutputIt, typename RedFunc>
 typename  std::enable_if<is_iterator<OutputIt>::value, void>::type
reduce (parallel_execution_tbb s, InputIt first, InputIt last, OutputIt firstOut, RedFunc const & reduce) {
    while( first != last ) {
       reduce(*first, *firstOut);
       first++;
       firstOut++;
    }
}


/*
template < typename InputIt, typename Output, typename RedFunc, typename FinalReduce>
 typename std::enable_if<!is_iterator<Output>::value, void>::type
Reduce(parallel_execution_thr p, InputIt first, InputIt last, Output & firstOut, RedFunc const & reduce, FinalReduce const & freduce) {

    std::vector<std::thread> tasks;
    int numElements = last - first;
    int elemperthr = numElements/CXX_NUM_THREADS;

    //local output
    std::vector<Output> out(CXX_NUM_THREADS);
     int i;
    //Create threads
    for(i=1;i<CXX_NUM_THREADS;i++){

      auto begin = first + (elemperthr * i);
      auto end = first + (elemperthr * (i+1));
      if(i == CXX_NUM_THREADS -1) end = last;

      tasks.push_back(
         std::thread( [&](InputIt begin, InputIt end, int tid){
               while( begin != end ) {
                       reduce(*begin, out[tid] );
                       begin++;
               }

         },
         begin, end, i
         )
      );

    }

    //Main thread
    auto end = first + elemperthr;
    while(first!=end){
         reduce(*first , out[0]);
         first++;
    }

    //Join threads
    for(int i=0;i<CXX_NUM_THREADS-1;i++){
      tasks[i].join();
    }

    auto it = out.begin();
    while( it!= out.end()){
       freduce(*it,firstOut);
       it++;   
    }      

}
*/


/*

template <typename InputIt, typename OutputIt, typename ... MoreIn, typename TaskFunc>
 void Reduce( InputIt first, InputIt last, OutputIt firstOut, TaskFunc const & taskf, MoreIn ... inputs ) {
    while( first != last ) {
        *firstOut = taskf( *first, *inputs ... );
        NextInputs( inputs... );
        first++;
        firstOut++;
    }
}
*/
}
#endif

#endif