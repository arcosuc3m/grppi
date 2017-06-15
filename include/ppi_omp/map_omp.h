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

#ifndef GRPPI_MAP_OMP_H
#define GRPPI_MAP_OMP_H

using namespace std;
namespace grppi{

template <typename GenFunc, typename TaskFunc>
 void map(parallel_execution_omp p, GenFunc const &in, TaskFunc const &taskf){
   //Create a queue per thread
  std::vector<unique_ptr<Queue<typename std::result_of<GenFunc()>::type >>> queues;
  for(int i =0; i<p.num_threads-1; i++) queues.push_back( unique_ptr<Queue<typename std::result_of<GenFunc()>::type >>(new Queue<typename std::result_of<GenFunc()>::type >(DEFAULT_SIZE,p.lockfree) ) );

   #pragma omp parallel
   {
     #pragma omp single nowait
     {
   //Create threads
     for(int i=0;i<p.num_threads-1;i++){
        #pragma omp task firstprivate(i)
        {
             int tid = i;
             typename std::result_of<GenFunc()>::type item;
             item = (*queues[tid]).pop();
             while( item ){
                taskf(item.value());
                item = (*queues[tid]).pop();
             }
        }
     }	
      //Generate elements
     while(1){
       int rr = 0;
       auto k = in();
       if(k.end){
           for(int i=0;i<p.num_threads-1;i++){
               (*queues[i]).push(k);
           }
           break;
       }
       (*queues[rr]).push(k);
       rr++;
       rr = (rr < p.num_threads -1) ? rr : 0; 
     }
     //Join threads
     #pragma omp taskwait
     }
   }
}

template <typename InputIt, typename OutputIt, typename TaskFunc>
 void map(parallel_execution_omp p, InputIt first, InputIt last, OutputIt firstOut, TaskFunc const &taskf){
   
  int numElements = last - first;

  int elemperthr = numElements/p.num_threads;

  #pragma omp parallel
  {
   #pragma omp single nowait
   {
    for(int i=1;i<p.num_threads;i++){
      


      #pragma omp task firstprivate(i)
      {
        auto begin = first + (elemperthr * i);
        auto end = first + (elemperthr * (i+1));
        if(i == p.num_threads -1 ) end = last;
        auto out = firstOut + (elemperthr * i);
        while(begin!=end){
          *out = taskf(*begin);
          begin++;
          out++;
        }
      }
     }
      //Map main threads
      auto beg =first;
      auto out = firstOut;
      auto end = first+elemperthr;
      while(beg!=end){
            *out = taskf(*beg);
            beg++;
            out++;
      }
      #pragma omp taskwait
    }
  }
}


template <typename InputIt, typename OutputIt, typename ... MoreIn, typename TaskFunc>
 void internal_map(parallel_execution_omp p, InputIt first, InputIt last, OutputIt firstOut,
                         TaskFunc const &taskf, int i, int elemperthr, MoreIn ... inputs){
        //Calculate local input and output iterator 
        auto begin = first + (elemperthr * i);
        auto end = first + (elemperthr * (i+1));
        if( i == p.num_threads-1) end = last;
        auto out = firstOut + (elemperthr * i);
        advance_iterators(elemperthr*i, inputs ...);
        while(begin!=end){
           *out = taskf(*begin, *inputs ...);
           advance_iterators(inputs ...);
           begin++;
           out++;
        }
}


template <typename InputIt, typename OutputIt, typename ... MoreIn, typename TaskFunc>
 void map(parallel_execution_omp p, InputIt first, InputIt last, OutputIt firstOut, TaskFunc const & taskf, MoreIn ... inputs){
   //Calculate number of elements per thread
   int numElements = last - first;
   int elemperthr = numElements/p.num_threads;

   //Create tasks
   #pragma omp parallel
   {
   #pragma omp single nowait
   {
     for(int i=1;i<p.num_threads;i++){

       #pragma omp task firstprivate(i)
       {
           internal_map(p, first, last, firstOut, taskf, i, elemperthr, inputs ...);
       }
      //End task
     }
     //Map main thread
     internal_map(p, first,last, firstOut, taskf, 0, elemperthr, inputs ...);

   //Join threads
   #pragma omp taskwait
   }
   }
}
}
#endif
