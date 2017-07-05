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

#ifndef GRPPI_MAPREDUCE_OMP_H
#define GRPPI_MAPREDUCE_OMP_H

#ifdef GRPPI_OMP

#include "../reduce.h"

namespace grppi{

template <typename InputIt, typename Transformer, typename IdentityType, typename Combiner>
IdentityType map_reduce ( parallel_execution_omp& p, InputIt first, InputIt last, Transformer &&  transform_op,  Combiner &&combine_op, IdentityType init){

    using namespace std;
    IdentityType out = init;
    std::vector<IdentityType> partialOuts(p.num_threads);
    #pragma omp parallel
    {
    #pragma omp single nowait
    {
    int numElements = last - first;
    int elemperthr = numElements/p.num_threads;
    sequential_execution s {};

    for(int i=1;i<p.num_threads;i++){    
       #pragma omp task firstprivate(i)
       {
          auto begin = first + (elemperthr * i);
          auto end = first + (elemperthr * (i+1));
          if(i == p.num_threads -1 ) end= last;
          partialOuts[i] = map_reduce(s, begin, end, std::forward<Transformer>(transform_op), std::forward<Combiner>(combine_op),partialOuts[i]);
       }
    }

    partialOuts[0] = map_reduce(s, first,( first+elemperthr ), std::forward<Transformer>(transform_op), std::forward<Combiner>(combine_op), partialOuts[0] );
    #pragma omp taskwait
    }
    }

    for(auto & map : partialOuts){
       out = combine_op(out, map);
    } 
    return out;
}

}
#endif
#endif
