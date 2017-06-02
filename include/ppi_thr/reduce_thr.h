/**
* @version		GrPPI v0.1
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
using namespace std;
namespace grppi{
//typename std::enable_if<!is_iterator<Output>::value, bool>::type,

template < typename InputIt, typename Output, typename ReduceOperator>
inline typename std::enable_if<!is_iterator<Output>::value, void>::type 
reduce(parallel_execution_thr& p, InputIt first, InputIt last, Output & firstOut, ReduceOperator op) {

    typename ReduceOperator::result_type identityVal = !op(false,true);


    std::vector<std::thread> tasks;
    int numElements = last - first;
    int elemperthr = numElements/p.num_threads;
    std::atomic<int> finishedTask(1); 
    //local output
    std::vector<Output> out(p.num_threads);
    //Create threads
    for(int i=1;i<p.num_threads;i++){
      
      auto begin = first + (elemperthr * i);
      auto end = first + (elemperthr * (i+1));
      if(i == p.num_threads -1) end = last;

      p.pool.ioService.post(boost::bind<void>(
           [&](InputIt begin, InputIt end, int tid){
               out[tid] = identityVal;
//               begin++;
               
               for( ; begin != end; begin++ ) {
                   out[tid] = op( out[tid], *begin );
                   //begin++;
               }
               finishedTask++;
            },
            std::move(begin), std::move(end), i
      ));
     /* 
      tasks.push_back(

         std::thread( [&](InputIt begin, InputIt end, int tid){
            out[tid] = *begin;
            begin++;
         	while( begin != end ) {
		 		out[tid] = op(*begin, out[tid] );
		 		begin++;
	        }
         }, 
         begin, end, i
         )   
      );
*/
    }   
   
    //Main thread
    auto end = first + elemperthr;
    out[0] = identityVal;
//    first++;
    while(first!=end){
         out[0] = op( out[0], *first);
         first++;
    }
    while(finishedTask.load()!=p.num_threads);//{std::cout<<finishedTask.load()<<" "<<p.num_threads<<std::endl; }

    //Join threads
   /* for(int i=0;i<p.num_threads-1;i++){
      tasks[i].join();
    }*/
  
    auto it = out.begin();
    while( it!= out.end()){
       firstOut = op( firstOut, *it);
       it++;
    }
}

/*
template < typename InputIt, typename Output, typename RedFunc, typename FinalReduce>
inline typename std::enable_if<!is_iterator<Output>::value, void>::type
Reduce(parallel_execution_thr p, InputIt first, InputIt last, Output & firstOut, RedFunc const & reduce, FinalReduce const & freduce) {

    std::vector<std::thread> tasks;
    int numElements = last - first;
    int elemperthr = numElements/p.num_threads;

    //local output
    std::vector<Output> out(p.num_threads);
    int i;
    //Create threads
    for(i=1;i<p.num_threads;i++){

      auto begin = first + (elemperthr * i);
      auto end = first + (elemperthr * (i+1));
      if(i == p.num_threads -1) end = last;

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
    for(int i=0;i<p.num_threads-1;i++){
      tasks[i].join();
    }

    auto it = out.begin();
    while( it!= out.end()){
       freduce(*it,firstOut);
       it++;   
    }      

}
*/


template < typename InputIt, typename OutputIt, typename  RedFunc>
inline typename  std::enable_if<is_iterator<OutputIt>::value, void>::type 
reduce (parallel_execution_thr &p, InputIt first, InputIt last, OutputIt firstOut, RedFunc const & reduce) {
    while( first != last ) {
       reduce( *first, *firstOut);
       first++;
       firstOut++;
    }
}




template < typename InputIt, typename ReduceOperator>
inline typename ReduceOperator::result_type
reduce(parallel_execution_thr &p, InputIt first, InputIt last, ReduceOperator op) {
    typename ReduceOperator::result_type identityVal = !op(false,true);

//    std::vector<std::thread> tasks;
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
      p.pool.ioService.post(boost::bind<void>(
           [&](InputIt begin, InputIt end, int tid){
               out[tid] = identityVal;
    //           begin++;
               for( ; begin != end; begin++ ) {
                   out[tid] = op(out[tid], *begin );
                   //begin++;
               }
               finishedTask++;
            },
            std::move(begin), std::move(end), i
      ));
/*
      tasks.push_back(
         std::thread( [&](InputIt begin, InputIt end, int tid){
            out[tid] = *begin;
            begin++;
            while( begin != end ) {
                out[tid] = op(*begin, out[tid] );
                begin++;
            }
         },
         begin, end, i
         )
      );
*/
    }
    //Main thread
    auto end = first + elemperthr;
    out[0] = identityVal;
//    first++;
    for(;first!=end;first++){
         out[0] = op( out[0], *first);
    }

    //Join threads
  //  for(int i=0;i<p.num_threads-1;i++){
  //    tasks[i].join();
  //  }
    while(finishedTask.load()!=p.num_threads);//{std::cout<<finishedTask.load()<<" "<<p.num_threads<<std::endl; }
    
    typename ReduceOperator::result_type outVal = out[0];
    for(unsigned int i = 1; i < out.size(); i++){
       outVal = op(outVal, out[i]);
    }
    return outVal;
}


/*

template <typename InputIt, typename OutputIt, typename ... MoreIn, typename TaskFunc>
inline void Reduce( InputIt first, InputIt last, OutputIt firstOut, TaskFunc const & taskf, MoreIn ... inputs ) {
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
