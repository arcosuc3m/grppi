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

#ifndef GRPPI_STREAM_REDUCE_THRUST_H
#define GRPPI_STREAM_REDUCE_THRUST_H

#ifdef __CUDACC__

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

using namespace std;
namespace grppi{
template <typename GenFunc, typename TaskFunc, typename ReduceFunc, typename OutputType, typename Policy>
inline void stream_reduce(parallel_execution_thrust_internal<Policy> p, GenFunc const &in, TaskFunc const &taskf, ReduceFunc const &red, OutputType &reduce_value ){

    typedef typename std::result_of<GenFunc()>::type::value_type::value_type inputtype;

    Queue<typename std::result_of<GenFunc()>::type> queue(DEFAULT_SIZE,p.lockfree);
    Queue<OutputType> end_queue(DEFAULT_SIZE,p.lockfree);

    std::vector<std::thread> tasks;

    //Create threads
    for( int i = 0; i < p.num_gpus; i++ ) {
        tasks.push_back(
            std::thread(
                [&]() {
                    typename std::result_of<GenFunc()>::type item;

                    cudaSetDevice(i);
                    item = queue.pop( );
                    while( item ) {
                        OutputType out;
                        thrust::device_vector< inputtype > d_vec_in(item.elem.begin(), item.elem.end());
                        out = thrust::reduce(p.policy, d_vec_in.begin(), d_vec_in.end(), 0, taskf);
                        end_queue.push( out ) 
                        item = queue.pop( );
                    }
                }
            )
        );
    }
	
    //Generate elements
    while( 1 ) {
        auto k = in();
        queue.push( k );
        if( k.end ) {
            for( int i = 1; i < p.num_gpus; i++ )
                queue.push( k );
            break;
        }
    }

    //Join threads
    for( int i = 0; i < p.num_gpus; i++ ) {
        tasks[ i ].join();
    }

    //Reduce
    OutputType item;
    while( (item = end_queue.pop()) )
        red( item, reduce_value  );

}
}
/*
template <typename TaskFunc, typename RedFunc>
ReduceObj<parallel_execution_thr,TaskFunc, RedFunc> StreamReduce(parallel_execution_thr p, TaskFunc && taskf, RedFunc && red){
   return ReduceObj<parallel_execution_thr, TaskFunc, RedFunc>(p,taskf, red);
}

*/

#endif

#endif
