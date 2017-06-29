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

#ifndef GRPPI_REDUCE_TBB_H
#define GRPPI_REDUCE_TBB_H

#ifdef GRPPI_TBB

#include <tbb/tbb.h>

namespace grppi{

// Empty sequences are currently not supported.
template < typename InputIt, typename Combiner>
typename std::iterator_traits<InputIt>::value_type
reduce(parallel_execution_tbb &p, InputIt first, InputIt last, typename std::iterator_traits<InputIt>::value_type init, Combiner && combine_op){

    auto identityVal = init;
   return tbb::parallel_reduce(tbb::blocked_range<InputIt>( first, last ), identityVal,
              [&](const tbb::blocked_range<InputIt> &r,typename std::iterator_traits<InputIt>::value_type  temp){
                 for(InputIt i=r.begin(); i!= r.end(); ++i){
                   temp = combine_op( temp, *i);
                 }
                 return temp;
              },
              [&](typename std::iterator_traits<InputIt>::value_type a, typename std::iterator_traits<InputIt>::value_type b) -> typename std::iterator_traits<InputIt>::value_type
              {
                a = combine_op(a,b);
                return a;
              }
          );
   
}

template < typename InputIt, typename Combiner>
typename std::result_of< Combiner(typename std::iterator_traits<InputIt>::value_type, typename std::iterator_traits<InputIt>::value_type) >::type
reduce(parallel_execution_tbb &p, InputIt first, InputIt last, Combiner &&combine_op){
   auto identityVal = !combine_op(false,true);
   return reduce(p, first, last, identityVal, std::forward<Combiner>(combine_op));
}

}
#endif

#endif
