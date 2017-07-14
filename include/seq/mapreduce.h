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

#ifndef GRPPI_SEQ_MAPREDUCE_H
#define GRPPI_SEQ_MAPREDUCE_H

#include "../reduce.h"

namespace grppi{

//Parallel STL like function
template <typename InputIt, typename Transformer, typename Identity, typename Combiner>
Identity map_reduce ( sequential_execution &, InputIt first, InputIt last, Identity identity, Transformer &&  transform_op, Combiner && combine_op){
    Identity out = identity;

    while(first != last){
       auto mappedValue = transform_op(*first);
       out = combine_op(out, mappedValue);
       first++;
    }

    return out;
}

}
#endif
