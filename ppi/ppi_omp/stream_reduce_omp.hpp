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

#ifndef PPI_STREAMREDUCE_OMP
#define PPI_STREAMREDUCE_OMP
#include "reduce_omp.hpp"
using namespace std;
namespace grppi{
template <typename GenFunc, typename TaskFunc, typename ReduceFunc, typename OutputType>
inline void StreamReduce(parallel_execution_omp p, GenFunc const &in, TaskFunc const & taskf, ReduceFunc const &red, OutputType &reduce_value ){
	
    Queue<typename std::result_of<GenFunc()>::type> queue(DEFAULT_SIZE, p.lockfree);
    Queue<optional<OutputType>> end_queue(DEFAULT_SIZE, p.lockfree);
    std::atomic<int> nend(0);
    std::vector<std::thread> tasks;
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            //Create threads
            for( int i = 0; i < p.num_threads; i++ ) {
                #pragma omp task shared(queue, end_queue)
                {
                     typename std::result_of<GenFunc()>::type item;
        	         item = queue.pop( ) ;
                     while( item ) {
                         auto out = taskf( item.value() );
                         end_queue.push( optional<OutputType>(out) );
                         item = queue.pop( );
                     }
                     nend++;
                     if(nend == p.num_threads)
                        end_queue.push( optional<OutputType>() );

                }
            } 
            #pragma omp task shared(end_queue)            
            {
                optional<OutputType> item;
                while((item = end_queue.pop( )) )
                    red( item.value(), reduce_value  );
            }
	
            //Generate elements
            while( 1 ) {
                 auto k = in();
                 queue.push( k );
                 if( k.end ) {
                    for( int i = 1; i < p.num_threads; i++ )
                        queue.push( k );
                    break;
                }
            }
            #pragma omp taskwait
        }
    }

    
}


template <typename GenFunc, typename ReduceOperator, typename SinkFunc>
inline void StreamReduce(parallel_execution_omp p, GenFunc const &in, int windowsize, int offset, ReduceOperator const & op, SinkFunc const &sink)
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
           auto reduceVal = Reduce(p, buffer.begin(), buffer.end(), op);
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



template <typename TaskFunc, typename RedFunc>
ReduceObj<parallel_execution_omp,TaskFunc, RedFunc> StreamReduce(parallel_execution_omp p, TaskFunc && taskf, RedFunc && red){
   return ReduceObj<parallel_execution_omp, TaskFunc, RedFunc>(p,taskf, red);
}
}
#endif
