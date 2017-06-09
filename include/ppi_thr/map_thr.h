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

#ifndef GRPPI_MAP_THR_H
#define GRPPI_MAP_THR_H
using namespace std;
namespace grppi{
template <typename GenFunc, typename TaskFunc>
inline void map(parallel_execution_thr& p, GenFunc const &in, TaskFunc const & taskf){
   std::vector<std::thread> tasks;
   //Create a queue per thread
  std::vector<shared_ptr<Queue<typename std::result_of<GenFunc()>::type >>> queues;
  for(int i =0; i<p.num_threads-1; i++) queues.push_back( shared_ptr<Queue<typename std::result_of<GenFunc()>::type >>(new Queue<typename std::result_of<GenFunc()>::type >(DEFAULT_SIZE,p.lockfree) ) );

   //Create threads
   for(int i=1;i<p.num_threads;i++){
      tasks.push_back(
         std::thread( [&](int tid){
            // Register the thread in the execution model
            p.register_thread();
            typename std::result_of<GenFunc()>::type item;
            item = (*queues[tid]).pop();
            while( item ){
              taskf(item.value());
              item = (*queues[tid]).pop();
            }
          
            // Deregister the thread in the execution model
            p.deregister_thread();
        }
        , (i-1)));
   }  
   //Generate elements
   while(1){
       int rr = 0;
       auto k = in();
       if(!k){
           for(int i=0;i<p.num_threads-1;i++){
               (*queues[i]).push(k);
           }
           break;
       }
      
       (*queues[rr]).push(k);
       rr++;
       rr = (rr < p.num_threads-1) ? rr : 0; 
   }
   //Join threads
   for(int i=0;i<p.num_threads-1;i++){
      tasks[i].join();
   }
}

template <typename InputIt, typename OutputIt, typename TaskFunc>
inline void map(parallel_execution_thr& p, InputIt first,InputIt last, OutputIt firstOut, TaskFunc const & taskf){
   
   std::vector<std::thread> tasks;
   int numElements = last - first; 
   int elemperthr = numElements/p.num_threads; 

   for(int i=1;i<p.num_threads;i++){
      auto begin = first + (elemperthr * i); 
      auto end = first + (elemperthr * (i+1)); 

      if(i == p.num_threads -1 ) end= last;

      auto out = firstOut + (elemperthr * i);
      tasks.push_back(
        std::thread( [&](InputIt begin, InputIt end, OutputIt out){
          // Register the thread in the execution model
          p.register_thread();
          
          while(begin!=end){
            *out = taskf(*begin);
            begin++;
            out++;
          }
          
          // Deregister the thread in the execution model
          p.deregister_thread();
        },
        begin, end, out)
      );
   }
   //Map main threads
   auto end = first+elemperthr;
   while(first!=end){
         *firstOut = taskf(*first);
         first++;
         firstOut++;
   }

   //Join threads
   for(int i=0;i<p.num_threads-1;i++){
      tasks[i].join();
   }

}





template <typename InputIt, typename OutputIt, typename ... MoreIn, typename TaskFunc>
inline void map(parallel_execution_thr& p, InputIt first, InputIt last, OutputIt firstOut, TaskFunc const & taskf, MoreIn ... inputs){
 std::vector<std::thread> tasks;
   //Calculate number of elements per thread
   int numElements = last - first;
   int elemperthr = numElements/p.num_threads;
   //Create tasks
   for(int i=1;i<p.num_threads;i++){
      //Calculate local input and output iterator 
      auto begin = first + (elemperthr * i);
      auto end = first + (elemperthr * (i+1));
      if( i == p.num_threads-1) end = last;
      auto out = firstOut + (elemperthr * i);
      //Begin task
      tasks.push_back(
        std::thread( [&](InputIt begin, InputIt end, OutputIt out, int tid, int nelem, MoreIn ... inputs){

            // Register the thread in the execution model
            p.register_thread();

            advance_iterators(nelem*tid, inputs ...);
            while(begin!=end){
               *out = taskf(*begin, *inputs ...);
               advance_iterators(inputs ...);
               begin++;
               out++;
            }

            // Deregister the thread in the execution model
            p.deregister_thread();
        },
        begin, end, out, i, elemperthr, inputs...)
      );
      //End task
   }
   //Map main thread
   auto end = first + elemperthr;
   while(first!=end){
         *firstOut = taskf(*first, *inputs ...);
         advance_iterators(inputs ...);
         first++;
         firstOut++;
   }


   //Join threads
   for(int i=0;i<p.num_threads-1;i++){
      tasks[i].join();
   }
}
}
#endif
