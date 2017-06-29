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
template < typename InputIt, typename OutputIt, typename Transformer, typename Combiner, typename ... MoreIn >
void map_reduce (sequential_execution &s, InputIt first, InputIt last, OutputIt firstOut, Transformer && transform_op, Combiner && combine_op, MoreIn ... inputs) {

    while( first != last ) {
       auto mapresult = transform_op(*first, inputs ... );
       *firstOut = reduce(s, mapresult.begin(), mapresult.end(), *firstOut, std::forward<Combiner>(combine_op));
       first++;
       firstOut++;
    }
}

//Parallel STL like function
template <typename InputIt, typename Transformer, class T, typename Combiner>
 T map_reduce ( sequential_execution, InputIt first, InputIt last, Transformer &&  transform_op, T init, Combiner && combine_op){
    T out = init;

    while(first != last){
       auto mappedValue = transform_op(*first);
       out = combine_op(out, mappedValue);
       first++;
    }

    return out;
}

template <typename InputIt, typename Transformer, class T, typename Combiner>
 T map_reduce ( sequential_execution, InputIt first, InputIt last, Transformer &&  transform_op, Combiner && combine_op){
    T out;  
    bool firstElement = true;
    while(first != last){
       auto mappedValue = transform_op(*first);
       if(firstElement) {
          firstElement = false;
          out = mappedValue;
       }else{
          out = combine_op(out, mappedValue);
       }
       first++;
    }

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
