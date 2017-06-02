/**
* @version		GrPPI v0.1
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

#ifndef GRPPI_FARM_SEQ_H
#define GRPPI_FARM_SEQ_H

using namespace std;
namespace grppi{
template <typename GenFunc, typename TaskFunc>
void farm(sequential_execution , GenFunc const &in, TaskFunc const & taskf ) {

    while( 1 ) {
        auto k = in();
        if( !k ) 
            break;
        taskf( k.value() );
    }
}

template <typename GenFunc, typename TaskFunc, typename SinkFunc>
void farm(sequential_execution , GenFunc const &in, TaskFunc const & taskf, SinkFunc const &sink ) {

    while( 1 ) {
        auto k = in();
        if( !k ) 
            break;
        auto r = taskf( k.value() );
        sink(r);
    }
}


template <typename TaskFunc>
FarmObj<sequential_execution,TaskFunc> farm(sequential_execution s, TaskFunc && taskf){
   return FarmObj<sequential_execution, TaskFunc>(s ,taskf);
}
}
#endif
