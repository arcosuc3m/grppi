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

#ifndef GRPPI_NATIVE_MAPREDUCE_H
#define GRPPI_NATIVE_MAPREDUCE_H

#include "parallel_execution_native.h"
#include "../reduce.h"


namespace grppi{

template <typename InputIt, typename Transformer, typename Identity, typename Combiner>
Identity map_reduce ( parallel_execution_native& p, InputIt first, InputIt last, Identity identity, Transformer && transform_op,  Combiner &&combine_op){

    using namespace std;
    Identity out = identity;
    std::vector<Identity> partialOuts(p.num_threads);
    std::vector<std::thread> tasks;
    int numElements = last - first;
    int elemperthr = numElements/p.num_threads;
    sequential_execution s {};

    for(int i=1;i<p.num_threads;i++){    
       auto begin = first + (elemperthr * i);
       auto end = first + (elemperthr * (i+1));
       if(i == p.num_threads -1 ) end= last;
       tasks.emplace_back([&](InputIt beg, InputIt en, int id) {
         // Register thread
         p.register_thread();

         partialOuts[id] = map_reduce(s, beg, en, partialOuts[id], std::forward<Transformer>(transform_op), std::forward<Combiner>(combine_op));

         // Deregister thread
         p.deregister_thread();
       }, begin, end, i);
    }

    partialOuts[0] = map_reduce(s, first,( first+elemperthr ), partialOuts[0], std::forward<Transformer>(transform_op), std::forward<Combiner>(combine_op));

    for(auto task = tasks.begin();task != tasks.end();task++){    
       (*task).join();
    }
    for(auto & map : partialOuts){
       out = combine_op(out, map);
    } 
    return out;
}

}
#endif
