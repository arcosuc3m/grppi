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

#ifndef GRPPI_REDUCE_OMP_H
#define GRPPI_REDUCE_OMP_H

#ifdef GRPPI_OMP

#include <thread>

namespace grppi{
//typename std::enable_if<!is_iterator<Output>::value, bool>::type,

template < typename InputIt, typename Output, typename ReduceOperator>
 typename std::enable_if<!is_iterator<Output>::value, void>::type 
reduce(parallel_execution_omp &p, InputIt first, InputIt last, Output &firstOut, ReduceOperator op) {
    int numElements = last - first;
    int elemperthr = numElements/p.num_threads;
    typename ReduceOperator::result_type identityVal = !op(false,true);
  
    //local output
    std::vector<Output> out(p.num_threads);
    #pragma omp parallel
    {
    #pragma omp single nowait
    {
    //Create threads
      for(int i=1;i<p.num_threads;i++){
        #pragma omp task firstprivate(i) shared(out) 
        { 
          auto begin = first + (elemperthr * i);
          auto end = first + (elemperthr * (i+1));
          out[i] = identityVal;
//          begin++;
          if(i == p.num_threads -1) end = last;
          while( begin < end ) {
	       out[i] = op( out[i], *begin );
	       begin++;
          }
        }
      } 

   
      //Main thread
      auto begin = first;
      auto end = first + elemperthr;
      out[0] = identityVal;
//      begin++;
      while(begin!=end){
         out[0] = op( out[0], *begin);
         begin++;
      }
    #pragma omp taskwait
    }
    }
     
    auto it = out.begin();
    while( it!= out.end()){
       firstOut = op( firstOut, *it);
       it++;
    }
}
/*
template < typename InputIt, typename Output, typename RedFunc, typename FinalReduce>
 typename std::enable_if<!is_iterator<Output>::value, void>::type
Reduce(parallel_execution_omp p, InputIt first, InputIt last, Output & firstOut, RedFunc && reduce, FinalReduce && freduce) {

    int numElements = last - first;
    int elemperthr = numElements/p.num_threads;
    //local output
    std::vector<Output> out(p.num_threads);
    #pragma omp parallel
    {
    #pragma omp single nowait
    {
      //Create threads
      for(int i=1;i<p.num_threads;i++){
        #pragma omp task firstprivate(i)
        {
          auto begin = first + (elemperthr * i);
          auto end = first + (elemperthr * (i+1));
          if(i == p.num_threads -1) end = last;
          while( begin != end ) {
            reduce(*begin, out[i] );
            begin++;
          }
        }
      }

      //Main thread
      auto end = first + elemperthr;
      while(first!=end){
         reduce(*first , out[0]);
         first++;
      }
    #pragma omp taskwait
    }
    }
    auto it = out.begin();
    while( it!= out.end()){
       freduce(*it,firstOut);
       it++;   
    }      

}
*/
template < typename InputIt, typename OutputIt, typename RedFunc>
 typename  std::enable_if<is_iterator<OutputIt>::value, void>::type
reduce(parallel_execution_omp &p, InputIt first, InputIt last, OutputIt firstOut, RedFunc &&reduce) {
    while( first != last ) {
       reduce(*first, *firstOut);
       first++;
       firstOut++;
    }
}


/*

template <typename InputIt, typename OutputIt, typename ... MoreIn, typename Operation>
 void Reduce( InputIt first, InputIt last, OutputIt firstOut, Operation && op, MoreIn ... inputs ) {
    while( first != last ) {
        *firstOut = op( *first, *inputs ... );
        advance_iterators( inputs... );
        first++;
        firstOut++;
    }
}
*/

template < typename InputIt, typename ReduceOperator>
 typename ReduceOperator::result_type
reduce(parallel_execution_omp &p, InputIt first, InputIt last, ReduceOperator op) {
    int numElements = last - first;
    int elemperthr = numElements/p.num_threads;
    typename ReduceOperator::result_type identityVal = !op(false,true);

    //local output
    std::vector<typename ReduceOperator::result_type> out(p.num_threads);
    //Create threads
    #pragma omp parallel
    {
    #pragma omp single nowait
    {
    for(int i=1;i<p.num_threads;i++){

      auto begin = first + (elemperthr * i);
      auto end = first + (elemperthr * (i+1));
      if(i == p.num_threads -1) end = last;

         #pragma omp task firstprivate (begin, end,i)
         {
            out[i] = identityVal;
    //        begin++;
            while( begin != end ) {
                out[i] = op(*begin, out[i] );
                begin++;
            }
         }

    }
    
    //Main thread
    auto end = first + elemperthr;
    out[0] = identityVal;
//    first++;
    while(first!=end){
         out[0] = op(*first , out[0]);
         first++;
    }

    #pragma omp taskwait
    }
    }

    typename ReduceOperator::result_type outVal = out[0];
    for(unsigned int i = 1; i < out.size(); i++){
       outVal = op(outVal, out[i]);
    }
    return outVal;
}
}

#endif

#endif