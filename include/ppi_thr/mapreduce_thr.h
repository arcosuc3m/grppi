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

#ifndef GRPPI_MAPREDUCE_THR_H
#define GRPPI_MAPREDUCE_THR_H

#include "../reduce.h"

namespace grppi{
template < typename InputIt, typename OutputIt, typename MapFunc, typename ReduceOperator,typename ... MoreIn >
 void map_reduce (parallel_execution_thr & p, InputIt first, InputIt last, OutputIt firstOut, MapFunc && map, ReduceOperator op, MoreIn ... inputs) {
    // Register the thread in the execution model
    p.register_thread();
    while( first != last ) {
       auto mapresult = map(*first, inputs ... );
       reduce(p, mapresult.begin(), mapresult.end(), *firstOut, op);
       first++;
       firstOut++;
    }
    // Deregister the thread in the execution model
    p.deregister_thread();
}


template <typename InputIt, typename MapFunc, class T, typename ReduceOperator>
 T map_reduce ( parallel_execution_thr& p, InputIt first, InputIt last, MapFunc &&  map, T init, ReduceOperator op){
    T out = init;
    std::vector<T> partialOuts(p.num_threads);
    std::vector<std::thread> tasks;
    int numElements = last - first;
    int elemperthr = numElements/p.num_threads;

    for(int i=1;i<p.num_threads;i++){    
       auto begin = first + (elemperthr * i);
       auto end = first + (elemperthr * (i+1));
       if(i == p.num_threads -1 ) end= last;
       tasks.push_back(
        std::thread( 
         [&](InputIt begin, InputIt end, T out){
            // Register thread
            p.register_thread();

            partialOuts[i] = MapReduce(sequential_execution{}, begin, last, std::forward<MapFunc>(map), op);

            // Deregister thread
            p.deregister_thread();
         },
         begin, end, out)
       );
    }
    partialOuts[0] = MapReduce(sequential_execution{}, begin, last, std::forward<MapFunc>(map), op);
    for(auto task = tasks.begin();task != tasks.end();task++){    
       (*task).join();
    }
    Reduce(sequential_execution{}, partialOuts.begin(), partialOuts.end(), out, op);

    return out;
}

/*

template <typename InputIt, typename OutputIt, typename ... MoreIn, typename TaskFunc>
 void Reduce( InputIt first, InputIt last, OutputIt firstOut, TaskFunc && taskf, MoreIn ... inputs ) {
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
