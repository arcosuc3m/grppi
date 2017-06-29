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
template < typename InputIt, typename OutputIt, typename Transformer, typename Combiner,typename ... MoreIn >
 void map_reduce (parallel_execution_native & p, InputIt first, InputIt last, OutputIt firstOut, Transformer && transform_op, Combiner && combine_op, MoreIn ... inputs) {

    // Register the thread in the execution model
    p.register_thread();
    while( first != last ) {
       auto mapresult = transform_op(*first, inputs ... );
       *firstOut = reduce( p, mapresult.begin(), mapresult.end(), std::forward<Combiner>(combine_op) );
       first++;
       firstOut++;
    }
    // Deregister the thread in the execution model
    p.deregister_thread();
}


template <typename InputIt, typename Transformer, class T, typename Combiner>
 T map_reduce ( parallel_execution_native& p, InputIt first, InputIt last, Transformer &&  transform_op, T init, Combiner &&combine_op){

    using namespace std;
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

            partialOuts[i] = MapReduce(sequential_execution{}, begin, last, std::forward<Transformer>(transform_op), std::forward<Combiner>(combine_op));

            // Deregister thread
            p.deregister_thread();
         },
         begin, end, out)
       );
    }
    partialOuts[0] = MapReduce(sequential_execution{}, begin, last, std::forward<Transformer>(transform_op), std::forward<Combiner>(combine_op));
    for(auto task = tasks.begin();task != tasks.end();task++){    
       (*task).join();
    }
    Reduce(sequential_execution{}, partialOuts.begin(), partialOuts.end(), out, std::forward<Combiner>(combine_op));

    return out;
}

/*

template <typename InputIt, typename OutputIt, typename ... MoreIn, typename Operation>
 void Reduce( InputIt first, InputIt last, OutputIt firstOut, Operation && op, MoreIn ... inputs ) {
    while( first != last ) {
        *firstOut = op( *first, *inputs ... );
        NextInputs( inputs... );
        first++;
        firstOut++;
    }
}
*/
}
#endif
