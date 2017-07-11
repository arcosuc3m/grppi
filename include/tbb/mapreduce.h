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

#ifndef GRPPI_TBB_MAPREDUCE_H
#define GRPPI_TBB_MAPREDUCE_H

#ifdef GRPPI_TBB

#include <tbb/tbb.h>

#include "../reduce.h"

namespace grppi{
template <typename InputIt, typename Transformer, typename T, typename Combiner>
 T map_reduce ( parallel_execution_tbb& p, InputIt first, InputIt last, Transformer && transform_op, Combiner && combine_op, T init){

    using namespace std;
    tbb::task_group g;

    T out = init;
    std::vector<T> partialOuts(p.num_threads);
    int numElements = last - first;
    int elemperthr = numElements/p.num_threads;
    sequential_execution s{};
    for(int i=1;i<p.num_threads;i++){    
       auto begin = first + (elemperthr * i);
       auto end = first + (elemperthr * (i+1));
       if(i == p.num_threads -1 ) end= last;
       g.run(
         [&, begin, end, i](){
            partialOuts[i] = map_reduce(s, begin, end, std::forward<Transformer>(transform_op), std::forward<Combiner>(combine_op), partialOuts[i] );
         }
         
       );
    }

    partialOuts[0] = map_reduce(s, first, (first+elemperthr), std::forward<Transformer>(transform_op), std::forward<Combiner>(combine_op), partialOuts[0] );
    g.wait();
    
    for( auto & res : partialOuts){
       out = combine_op(out, res);
    }
    return out;
}

template <typename InputIt, typename Transformer, typename Combiner>
auto map_reduce ( parallel_execution_tbb& p, InputIt first, InputIt last, Transformer &&  transform_op,  Combiner &&combine_op){

    typename std::result_of<Combiner(
    typename std::result_of<Transformer(typename std::iterator_traits<InputIt>::value_type)>::type,
    typename std::result_of<Transformer(typename std::iterator_traits<InputIt>::value_type)>::type)>::type init;

    return map_reduce ( p, first, last, std::forward<Transformer>( transform_op ),  std::forward<Combiner>( combine_op ), init);
}



}
#endif

#endif
