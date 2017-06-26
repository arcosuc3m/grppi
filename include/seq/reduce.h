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

#ifndef GRPPI_REDUCE_SEQ_H
#define GRPPI_REDUCE_SEQ_H

namespace grppi{


template < typename InputIt, typename ReduceOperator>
typename std::iterator_traits<InputIt>::value_type
reduce(sequential_execution &p, InputIt first, InputIt last, typename std::iterator_traits<InputIt>::value_type init, ReduceOperator &&op){
    auto firstOut = init;
    while( first != last ) {
       firstOut = op( firstOut, *first );
       first++;
    }
    return firstOut;
}

template < typename InputIt, typename ReduceOperator>
typename std::result_of< ReduceOperator(typename std::iterator_traits<InputIt>::value_type, typename std::iterator_traits<InputIt>::value_type) >::type
reduce(sequential_execution &p, InputIt first, InputIt last, ReduceOperator &&op){
   auto identityVal = !op(false,true);
   return reduce(p, first, last, identityVal, std::forward<ReduceOperator>(op));
}



}
#endif
