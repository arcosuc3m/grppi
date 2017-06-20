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

#ifndef GRPPI_STREAM_REDUCE_THR_H
#define GRPPI_STREAM_REDUCE_THR_H

#include "../reduce.h"

#include <thread>

namespace grppi{
template <typename GenFunc, typename Operation, typename ReduceFunc, typename OutputType>
 void stream_reduce(parallel_execution_native &p, GenFunc &&in, Operation &&op, ReduceFunc &&red, OutputType &reduce_value ){

    Queue<typename std::result_of<GenFunc()>::type> queue(DEFAULT_SIZE,p.lockfree);
    Queue<optional<OutputType>> end_queue(DEFAULT_SIZE,p.lockfree);
    std::atomic<int> nend (0);
    std::vector<std::thread> tasks;
    //Create threads
    for( int i = 0; i < p.num_threads; i++ ) {
        tasks.push_back(
            std::thread(
                [&]() {
                    // Register the thread in the execution model
                    p.register_thread();

                    typename std::result_of<GenFunc()>::type item;
                    item = queue.pop();
                    while( item ) {
                        OutputType out = op( item.value() );
                        end_queue.push( optional<OutputType>(out) ) ;
                        item = queue.pop();
                    }
                    nend++;
                    if(nend == p.num_threads)
                        end_queue.push(optional<OutputType>()); 
                    
                    // Deregister the thread in the execution model
                    p.deregister_thread();
                }
            )
        );
    }
    std::thread merge([&](){	
        optional<OutputType> item;
        while( (item = end_queue.pop( )) )
            red( item.value(), reduce_value  );
    });
    //Generate elements
    while( 1 ) {
        auto k = in();
        queue.push( k ) ;
        if( k.end ) {
            for( int i = 0; i < p.num_threads; i++ )
                queue.push( k ) ;
            break;
        }
    }


    //Join threads
    for( int i = 0; i < p.num_threads; i++ ) {
        tasks[ i ].join();
    }
    merge.join();
    //Reduce

}


template <typename GenFunc, typename ReduceOperator, typename SinkFunc>
 void stream_reduce(parallel_execution_native &p, GenFunc &&in, int windowsize, int offset, ReduceOperator && op, SinkFunc &&sink)
{
     
     std::vector<typename std::result_of<GenFunc()>::type::value_type> buffer;
     auto k = in();
     while(1){
        //Create a vector as a buffer 
        //If its not the las element and the window is not complete keep getting more elements
        while( k && buffer.size() != windowsize){
           buffer.push_back(k.value());
           k = in();
        }
        if(buffer.size()>0){
           //Apply the reduce function to the elements on the window
           auto reduceVal = reduce(p, buffer.begin(), buffer.end(), op);
           //Call to sink function
           sink(reduceVal);
           //Remove elements
           if(k){
              buffer.erase(buffer.begin(), buffer.begin() + offset);
           }
        }
        //If there is no more elements finallize the pattern
        if( !k ){
           break;
        }
    }

}



template <typename Operation, typename RedFunc>
reduction_info<parallel_execution_native,Operation, RedFunc> stream_reduce(parallel_execution_native p, Operation && op, RedFunc && red){
   return reduction_info<parallel_execution_native, Operation, RedFunc>(p,op, red);
}
}
#endif
