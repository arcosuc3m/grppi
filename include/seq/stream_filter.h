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

#ifndef GRPPI_STREAM_FILTER_SEQ_H
#define GRPPI_STREAM_FILTER_SEQ_H

namespace grppi{
template <typename GenFunc, typename FilterFunc, typename OutFunc>
void stream_filter(sequential_execution, GenFunc && in, FilterFunc && filter, OutFunc && out ) {

    while( 1 ) {
        auto k = in();
        if( k.end )
            break;
        if(filter(k.elem))
            out(k.elem);
    }
}


template <typename GenFunc, typename FilterFunc, typename OutFunc>
void stream_filter( GenFunc && in, FilterFunc && filter, OutFunc && out ) {

    while( 1 ) {
        auto k = in();
        if( k.end ) 
            break;
        if(filter(k.elem))
            out(k.elem);
    }
}

template <typename FilterFunc>
FilterObj<sequential_execution, FilterFunc> stream_filter(sequential_execution &s, FilterFunc && op){
   return FilterObj<sequential_execution, FilterFunc>(s, op);
}

}
#endif
