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

#ifndef GRPPI_STENCIL_SEQ_H
#define GRPPI_STENCIL_SEQ_H

namespace grppi{
template <typename InputIt, typename OutputIt, typename Operation, typename NFunc>
 void stencil(sequential_execution &s, InputIt first, InputIt last, OutputIt firstOut, Operation && op, NFunc && neighbor ) {
    while( first != last ) {
       auto neighbors = neighbor(first);   
       *firstOut = op(first, neighbors);
       first++;
       firstOut++;

    }
    
}


template <typename InputIt, typename OutputIt, typename ... MoreIn, typename Operation, typename NFunc>
 void stencil(sequential_execution &s, InputIt first, InputIt last, OutputIt firstOut, Operation && op, NFunc && neighbor, MoreIn ... inputs ) {
    while( first != last ) {
        auto neighbors = neighbor(first);
        *firstOut = op(first, neighbors, *inputs ...);
        NextInputs( inputs... );
        first++;
        firstOut++;
    }
}
}
#endif
