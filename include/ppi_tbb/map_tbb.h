/**
* @version    GrPPI v0.2
* @copyright    Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license    GNU/GPL, see LICENSE.txt
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

#ifndef GRPPI_MAP_TBB_H
#define GRPPI_MAP_TBB_H

#include <tbb/tbb.h>
namespace grppi{
using namespace std;
template <typename InputIt, typename OutputIt, typename TaskFunc>
void map(parallel_execution_tbb p, InputIt first,InputIt last, OutputIt firstOut, TaskFunc const & taskf){
   tbb::parallel_for(static_cast<std::size_t>(0),static_cast<std::size_t>( (last-first) ), [&] (std::size_t index){
           auto current = (firstOut+index);
           *current = taskf(*(first+index));
       }
   );   
}



template <typename InputIt, typename OutputIt, typename ... MoreIn, typename TaskFunc>
void map(parallel_execution_tbb p, InputIt first, InputIt last, OutputIt firstOut, TaskFunc const & taskf, MoreIn ... inputs){
   tbb::parallel_for(static_cast<std::size_t>(0),static_cast<std::size_t>( (last-first) ), [&] (std::size_t index){
           auto current = (firstOut+index);
           *current = taskf(*(first+index));
       }

   );   

}
}
#endif
