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

#ifndef PPI_MAP_THRUST
#define PPI_MAP_THRUST

#ifdef __CUDACC__

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/transform.h>

using namespace std;
namespace grppi{
template <typename GenFunc, typename TaskFunc, typename Policy>
inline void Map(parallel_execution_thrust_internal<Policy> p, GenFunc const &in, TaskFunc const & taskf){
   typedef typename std::result_of<GenFunc()>::type inputtype;
   std::vector<std::thread> tasks;
   //Create a queue per thread
   std::vector< 
        boost::lockfree::spsc_queue<
            inputtype
            , boost::lockfree::capacity<BOOST_QUEUE_SIZE>
        >
   > queues(p.num_gpus);
 
	
   //Create gpus
   for(int i=0;i<p.num_gpus;i++){
      tasks.push_back(
         std::thread( [&](int tid){
             typename std::result_of<GenFunc()>::type item;
             
             cudaSetDevice(i);
             while(!queues[tid].pop(item));
             while(item){
                thrust::device_vector< inputtype > d_vec_in(item.elem.begin(), item.elem.end());
                thrust::device_vector< inputtype  > d_vec_out(item.elem.size());
                thrust::transform(p.policy, d_vec_in.begin(), d_vec_in.end(), d_vec_out.begin(), taskf);
                thrust::copy(d_vec_out.begin(), d_vec_out.end(), item.elem.begin());
                while(!queues[tid].pop(item));
             }
        }
        , i));
   }	
   //Generate elements
   while(1){
       int rr = 0;
       auto k = in();
       if( !k ){
           for(int i=0;i<p.num_gpus;i++){
               while(!queues[i].push(k));
           }
           break;
       }
       while(!queues[rr].push(k));
       rr++;
       rr = (rr >= p.num_gpus) ? 0 : rr; 
   }
   //Join gpus
   for(int i=0;i<p.num_gpus;i++){
      tasks[i].join();
   }
}

template <typename InputIt, typename OutputIt, typename TaskFunc, typename Policy>
inline void Map(parallel_execution_thrust_internal<Policy>  p, InputIt first,InputIt last, OutputIt firstOut, TaskFunc const & taskf){
   
   std::vector<std::thread> tasks;

   int numElements = last - first;
   int elemperthr = numElements/p.num_gpus;

   for(int i=1;i<p.num_gpus;i++){
      auto begin = first + (elemperthr * i);
      auto end = first + (elemperthr * (i+1));

      if(i == p.num_gpus -1 ) end= last;

      auto out = firstOut + (elemperthr * i);
      tasks.push_back(
        std::thread( [&](InputIt begin, InputIt end, OutputIt out){
               cudaSetDevice(i); 
               thrust::device_vector< typename InputIt::value_type > d_vec_in(begin, end);
               thrust::device_vector< typename OutputIt::value_type > d_vec_out(end - begin);
               thrust::transform(p.policy, d_vec_in.begin(), d_vec_in.end(), d_vec_out.begin(), taskf);
               thrust::copy(d_vec_out.begin(), d_vec_out.end(), out);
        },
        begin, end, out)
      );
   }
   //Map main gpus
   cudaSetDevice(0);
   thrust::device_vector< typename InputIt::value_type > d_vec_in(first, first+elemperthr);
   thrust::device_vector< typename OutputIt::value_type > d_vec_out(elemperthr);

   thrust::transform(p.policy, d_vec_in.begin(), d_vec_in.end(), d_vec_out.begin(), taskf);

   thrust::copy(d_vec_out.begin(), d_vec_out.end(), firstOut);

   //Join gpus
   for(int i=0;i<p.num_gpus-1;i++){
      tasks[i].join();
   }

}


template <typename InputIt, typename OutputIt, typename ... MoreIn, typename TaskFunc, typename Policy>
inline void Map(parallel_execution_thrust_internal<Policy>  p, InputIt first, InputIt last, OutputIt firstOut, TaskFunc const & taskf, MoreIn ... inputs){
 std::vector<std::thread> tasks;
   //Calculate number of elements per thread
   int numElements = last - first;
   int elemperthr = numElements/p.num_gpus;
   //Create tasks
   for(int i=1;i<p.num_gpus;i++){
      //Calculate local input and output iterator 
      auto begin = first + (elemperthr * i);
      auto end = first + (elemperthr * (i+1));
      if( i == p.num_gpus-1) end = last;
      auto out = firstOut + (elemperthr * i);
      //Begin task
      tasks.push_back(
        std::thread( [&](InputIt begin, InputIt end, OutputIt out, int tid, int nelem, MoreIn ... inputs){
            GetStart(nelem, tid, inputs ...);
            while(begin!=end){
               *out = taskf(*begin, *inputs ...);
	           NextInputs(inputs ...);
               begin++;
               out++;
            }
        },
        begin, end, out, i, elemperthr, inputs...)
      );
      //End task
   }
   //Map main thread
   auto end = first + elemperthr;
   while(first!=end){
         *firstOut = taskf(*first, *inputs ...);
         NextInputs(inputs ...);
         first++;
         firstOut++;
   }


   //Join gpus
   for(int i=0;i<p.num_gpus-1;i++){
      tasks[i].join();
   }
}
}
#endif

#endif
