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

#ifndef GRPPI_MAPREDUCE_SEQ_H
#define GRPPI_MAPREDUCE_SEQ_H

#include "../reduce.h"

namespace grppi{
template < typename InputIt, typename OutputIt, typename MapFunc, typename ReduceOperator, typename ... MoreIn >
void map_reduce (sequential_execution const &s, InputIt first, InputIt last, OutputIt firstOut, MapFunc const & map, ReduceOperator op, MoreIn ... inputs) {
    while( first != last ) {
       auto mapresult = map(*first, inputs ... );
       reduce(s, mapresult.begin(), mapresult.end(), *firstOut, op);
       first++;
       firstOut++;
    }
}

//Parallel STL like function
template <typename InputIt, typename MapFunc, class T, typename ReduceOperator>
 T map_reduce ( sequential_execution, InputIt first, InputIt last, MapFunc const &  map, T init, ReduceOperator op){
    T out = init;

    while(first != last){
       auto mappedValue = map(*first);
       out = op(out, mappedValue);
       first++;
    }

    return out;
}

template <typename InputIt, typename MapFunc, class T, typename ReduceOperator>
 T map_reduce ( sequential_execution, InputIt first, InputIt last, MapFunc const &  map, ReduceOperator op){
    T out;  
    bool firstElement = true;
    while(first != last){
       auto mappedValue = map(*first);
       if(firstElement) {
          firstElement = false;
          out = mappedValue;
       }else{
          out = op(out, mappedValue);
       }
       first++;
    }

}
/*

template <typename InputIt, typename OutputIt, typename ... MoreIn, typename TaskFunc>
 void Reduce( InputIt first, InputIt last, OutputIt firstOut, TaskFunc const & taskf, MoreIn ... inputs ) {
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
