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


#ifndef GRPPI_OMP_STREAM_REDUCE_H
#define GRPPI_OMP_STREAM_REDUCE_H

#ifdef GRPPI_OMP

#include "../reduce.h"
#include "parallel_execution_omp.h"


namespace grppi{

template <typename Generator, typename Combiner, typename Consumer, typename Identity>
 void stream_reduce(parallel_execution_omp &p, Generator &&gen, int windowsize, int offset, Combiner && comb, Consumer &&cons, Identity identity)
{

     std::vector<typename std::result_of<Generator()>::type::value_type> buffer;
     auto k = gen();
     while(1){
        //Create a vector as a buffer 
        //If its not the las element and the window is not complete keep getting more elements
        while( k && buffer.size() != windowsize){
           buffer.push_back(k.value());
           k = gen();
        }
        if(buffer.size()>0){
           //Apply the reduce function to the elements on the window
           auto reduceVal = reduce(p, buffer.begin(), buffer.end(), identity, std::forward<Combiner>(comb) );
           //Call to sink function
           cons(reduceVal);
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
reduction_info<parallel_execution_omp,Operation, RedFunc> stream_reduce(parallel_execution_omp &p, Operation && op, RedFunc && red){
   return reduction_info<parallel_execution_omp, Operation, RedFunc>(p,op, red);
}
}
#endif

#endif
