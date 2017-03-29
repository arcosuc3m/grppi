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

#ifndef PPI_FARM_THRUST
#define PPI_FARM_THRUST

#ifdef __CUDACC__

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <typeinfo>

using namespace std;
namespace grppi{
/*
template <typename T, typename in, typename out>
class FarmObj{
    const T task;
    parallel_execution_thr exec;
    public:
        FarmObj(parallel_execution_thr p,T && farm){
            task=farm;
            exec=p;
        } 

}
*/

template <typename GenFunc, typename TaskFunc, typename SinkFunc, typename Policy>
inline void Farm(parallel_execution_thrust_internal<Policy> p, GenFunc const &in, TaskFunc const & taskf , SinkFunc const &sink) {

    //Create gpus
    std::vector<std::thread> tasks;
    Queue< typename std::result_of<GenFunc()>::type > queuein(DEAFULT_SIZE,p.lockfree);
    typedef typename std::result_of<GenFunc()>::type::value_type::value_type inputtype;
    typedef typename std::result_of<TaskFunc(inputtype)>::type outtype;
    
    // FIXME: __device__ macro does not capture. Output type must be equal to input type.
    Queue< optional < std::vector < inputtype  > > > queueout(DEFAULT_SIZE,p.lockfree);

    for( int i = 0; i < p.num_gpus; i++ ) {
        tasks.push_back(
            std::thread(
                [&](){
                    typename std::result_of<GenFunc()>::type item;
                    cudaSetDevice(i);  
                    item = queuein.pop();
                    while( item ) {
                       thrust::device_vector< inputtype > d_vec_in(item.elem.begin(), item.elem.end());
    	               // FIXME: __device__ macro does not capture. Output type must be equal to input type.
                       thrust::device_vector< inputtype > d_vec_out(item.elem.size());
                       // FIXME: __device__ macro does not capture. Output type must be equal to input type.
                       std::vector < inputtype > out(item.elem.size());

                       thrust::transform(p.policy, d_vec_in.begin(), d_vec_in.end(), d_vec_out.begin(), taskf);

                       thrust::copy(d_vec_out.begin(), d_vec_out.end(), out.begin());
                       queueout.push( optional< std::vector < inputtype >>(out) ) );
                       item = queuein.pop();
                    }
                    queueout.push( optional< std::vector< inputtype >>() );


                }
            )
        );
    }

    //SINK 

    tasks.push_back(
         std::thread(
            [&](){
                 //optional< std::vector < typename std::result_of<TaskFunc(typename std::result_of<GenFunc()>::type::value_type::value_type)>::type> > item;
		         optional< std::vector < inputtype >> item;
                 item = queueout.pop( );
                 while( item ) {
                    sink( item.value() );
                    item = queueout.pop( );
                 }
             }
        )
    );

   //Generate elements
    while( 1 ) {
        auto k = in();
        queuein.push( k );
        if( k.end ) {
            for( int i = 1; i < p.num_gpus; i++ ) {
                queuein.push( k );
            }
            break;
        }
    }

    //Join gpus
    for( int i = 0; i < tasks.size(); i++ )
       tasks[ i ].join();
}
	
template <typename GenFunc, typename TaskFunc, typename Policy>
inline void Farm(parallel_execution_thrust_internal<Policy> p, GenFunc const &in, TaskFunc const & taskf ) {

    typedef typename std::result_of<GenFunc()>::type::value_type::value_type inputtype;
    std::vector<std::thread> tasks;
    Queue< typename std::result_of<GenFunc()>::type > queue(DEFAULT_SIZE,p.lockfree);
    //Create gpus
    for( int i = 0; i < p.num_gpus; i++ ) {
        tasks.push_back(
            std::thread( 
                [&](){
                    typename std::result_of<GenFunc()>::type item;
                    cudaSetDevice(i);
                    item = queue.pop( );
                    while( item ) {
                       thrust::device_vector< inputtype > d_vec_in(item.elem.begin(), item.elem.end());
                       thrust::for_each(p.policy, d_vec_in.begin(), d_vec_in.end(),  taskf);
                       thrust::copy(d_vec_in.begin(), d_vec_in.end(), item.elem.begin());
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
            for( int i = 1; i < p.num_gpus; i++ ) {
                queue.push( k );
            }
            break;
        }
    }

    //Join gpus
    for( int i = 0; i < p.num_gpus; i++ )
       tasks[ i ].join();
}

/*
template <typename TaskFunc, typename Policy>
FarmObj<parallel_execution_thrust<Policy>,TaskFunc> Farm(parallel_execution_thrust<Policy> p, TaskFunc && taskf){
   return FarmObj<parallel_execution_thrust<Policy>, TaskFunc>(p,taskf);
}
*/
}
#endif

#endif
