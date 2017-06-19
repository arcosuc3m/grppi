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

#ifndef GRPPI_STREAM_REDUCE_SEQ_H
#define GRPPI_STREAM_REDUCE_SEQ_H

#include "../reduce.h"
#include <vector>

namespace grppi{
template <typename GenFunc, typename TaskFunc, typename ReduceFunc, typename OutputType>
 void stream_reduce(sequential_execution &s, GenFunc &&in, TaskFunc && taskf, ReduceFunc &&red, OutputType &reduce_value ) {

    while( 1 ) {
        auto k = in();
        if( !k ) 
            break;
        auto u = taskf(k.value());
        red(u, reduce_value);
    }
}


template <typename GenFunc, typename TaskFunc, typename ReduceFunc>
 void stream_reduce(sequential_execution &s, GenFunc &&in, TaskFunc && taskf, ReduceFunc &&red) {
    while( 1 ) {
        auto k = in();
        if( !k )
            break;
        auto u = taskf(k.value());
        red(u);
    }
}


template <typename GenFunc, typename ReduceOperator, typename SinkFunc>
 void stream_reduce(sequential_execution &s, GenFunc &&in, int windowsize, int offset, ReduceOperator && op, SinkFunc &&sink)
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
           auto reduceVal = reduce(s, buffer.begin(), buffer.end(), op);
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
}
#endif
