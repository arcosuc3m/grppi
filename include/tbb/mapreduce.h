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

#ifndef GRPPI_MAPREDUCE_TBB_H
#define GRPPI_MAPREDUCE_TBB_H

#ifdef GRPPI_TBB

#include <tbb/tbb.h>

#include "../reduce.h"

namespace grppi{
template <typename InputIt, typename Transformer, typename IdentityType, typename Combiner>
IdentityType map_reduce ( parallel_execution_tbb& p, InputIt first, InputIt last, IdentityType init, Transformer && transform_op, Combiner && combine_op){

    using namespace std;
    tbb::task_group g;

    IdentityType out = init;
    std::vector<IdentityType> partialOuts(p.num_threads);
    int numElements = last - first;
    int elemperthr = numElements/p.num_threads;
    sequential_execution s{};
    for(int i=1;i<p.num_threads;i++){    
       auto begin = first + (elemperthr * i);
       auto end = first + (elemperthr * (i+1));
       if(i == p.num_threads -1 ) end= last;
       g.run(
         [&, begin, end, i](){
            partialOuts[i] = map_reduce(s, begin, end, partialOuts[i], std::forward<Transformer>(transform_op), std::forward<Combiner>(combine_op));
         }
         
       );
    }

    partialOuts[0] = map_reduce(s, first, (first+elemperthr), partialOuts[0], std::forward<Transformer>(transform_op), std::forward<Combiner>(combine_op));
    g.wait();
    
    for( auto & res : partialOuts){
       out = combine_op(out, res);
    }
    return out;
}

}
#endif

#endif
