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
#ifndef GRPPI_SEQ_STREAM_ITERATION_H
#define GRPPI_SEQ_STREAM_ITERATION_H

#include "sequential_execution.h"

namespace grppi{

template<typename GenFunc, typename Operation, typename Predicate, typename OutFunc>
 void stream_iteration(sequential_execution, GenFunc && in, Operation && f, Predicate && condition, OutFunc && out){
   while(1){
       auto k = in();
       if(!k) break;
       auto val = *k;
       do{
         val = f(val);
       }while(condition(val));
       out(val);
   }
}


template<typename GenFunc, typename Operation, typename Predicate, typename OutFunc>
 void stream_iteration(sequential_execution &s, GenFunc && in, farm_info<sequential_execution, Operation> & f, Predicate && condition, OutFunc && out){
    stream_iteration(s, std::forward<GenFunc>( in ), std::forward<farm_info<sequential_execution, Operation> &&>( f ), 
          std::forward<Predicate>( condition), std::forward< OutFunc >( out ) );
}

template<typename GenFunc, typename Operation, typename Predicate, typename OutFunc>
 void stream_iteration(sequential_execution, GenFunc && in, farm_info<sequential_execution, Operation> && f, Predicate && condition, OutFunc && out){
   while(1){
       auto k = in();       
       if(!k) break;
       auto val = k.value();
       do{
          val = f.task(val);
       } while(condition(val));
       out(val);
   }
}

template<typename GenFunc, typename Predicate, typename OutFunc, typename ...Stages>
 void stream_iteration(sequential_execution &s, GenFunc && in, pipeline_info<sequential_execution, Stages...> & f, Predicate && condition, OutFunc && out){
    stream_iteration(s, std::forward<GenFunc>(in), std::forward<pipeline_info<sequential_execution, Stages...> &&>( f ), std::forward<Predicate>(condition), std::forward< OutFunc>( out ));
}

template<typename GenFunc, typename Predicate, typename OutFunc, typename ...Stages>
 void stream_iteration(sequential_execution &s, GenFunc && in, pipeline_info<sequential_execution, Stages...> && f, Predicate && condition, OutFunc && out){
   while(1){
       auto k = in();
       if(!k) break; 
       auto val = k.value();
       do{
             val = composed_pipeline<typename std::result_of<GenFunc()>::type::value_type,0,Stages...>(val,std::forward<pipeline_info<sequential_execution, Stages...>>(f) );
       }while(condition(val));
       out(val);
   }
  
}
}

#endif
