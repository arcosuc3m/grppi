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

#ifndef PPI_REDUCE_THRUST
#define PPI_REDUCE_THRUST

#ifdef __CUDACC__

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>

using namespace std;
namespace grppi{

//typename std::enable_if<!is_iterator<Output>::value, bool>::type,

template < typename InputIt, typename Output, typename RedFunc, typename Policy, typename ReduceOperator>
inline typename std::enable_if<!is_iterator<Output>::value, void>::type 
Reduce(parallel_execution_thrust_internal<Policy>  p, InputIt first, InputIt last, Output & firstOut, RedFunc const & reduce, ReduceOperator op) {
    std::vector<std::thread> tasks;
    int numElements = last - first;
    int elemperthr = numElements/p.num_gpus;
    
    //local output
    std::vector<Output> out(p.num_gpus, op.init());
    //Create threads
    for(int i=1;i<p.num_gpus;i++){
      
      auto begin = first + (elemperthr * i);
      auto end = first + (elemperthr * (i+1));
      if(i == p.num_gpus -1) end = last;
      
      tasks.push_back(
         std::thread( [&](InputIt begin, InputIt end, int tid){
              cudaSetDevice(i);
              thrust::device_vector< typename InputIt::value_type > d_vec_in(begin, end);
              out[tid] = thrust::reduce(p.policy, d_vec_in.begin(), d_vec_in.end(), 0, reduce);

         }, 
         begin, end, i
         )
      );

    }   
   
    //Main thread
    cudaSetDevice(0);
    thrust::device_vector< typename InputIt::value_type > d_vec_in_local(first, first + elemperthr);
    out[0] = thrust::reduce(p.policy, d_vec_in_local.begin(), d_vec_in_local.end(), 0, reduce);         

    //Join threads
    for(int i=0;i<p.num_gpus-1;i++){
      tasks[i].join();
    }
  
    thrust::device_vector< typename InputIt::value_type > d_vec_in(out.begin(), out.end());
    firstOut = thrust::reduce(p.policy, d_vec_in.begin(), d_vec_in.end(), 0, reduce);         
}

/*
template < typename InputIt, typename Output, typename RedFunc, typename FinalReduce, typename Policy>
inline typename std::enable_if<!is_iterator<Output>::value, void>::type
Reduce(parallel_execution_thrust_internal<Policy>  p, InputIt first, InputIt last, Output & firstOut, RedFunc const & reduce, FinalReduce const & freduce) {

    std::vector<std::thread> tasks;
    int numElements = last - first;
    int elemperthr = numElements/p.num_gpus;

    //local output
    std::vector<Output> out(p.num_gpus);
     int i;
    //Create threads
    for(i=1;i<p.num_gpus;i++){

      auto begin = first + (elemperthr * i);
      auto end = first + (elemperthr * (i+1));
      if(i == p.num_gpus -1) end = last;

      tasks.push_back(
         std::thread( [&](InputIt begin, InputIt end, int tid){
                  cudaSetDevice(i);
                  thrust::device_vector< typename InputIt::value_type > d_vec_in(begin, end);
                  out[tid] = thrust::reduce(p.policy, d_vec_in.begin(), d_vec_in.end(), 0, reduce);

         },
         begin, end, i
         )
      );

    }

    //Main thread
    cudaSetDevice(0);
    thrust::device_vector< typename InputIt::value_type > d_vec_in_local(first, first + elemperthr);
    out[0] = thrust::reduce(p.policy, d_vec_in_local.begin(), d_vec_in_local.end(), 0, reduce);

    //Join threads
    for(int i=0;i<p.num_gpus-1;i++){
      tasks[i].join();
    }

    thrust::device_vector< typename InputIt::value_type > d_vec_in(out.being(), out.end());
    firstOut = thrust::reduce(p.policy, d_vec_in.begin(), d_vec_in.end(), 0, freduce);

}
*/

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

#endif
