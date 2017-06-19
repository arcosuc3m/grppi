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

#ifndef GRPPI_STENCIL_TBB_H
#define GRPPI_STENCIL_TBB_H

#ifdef GRPPI_TBB

#include <tbb/tbb.h>

namespace grppi{
template <typename InputIt, typename OutputIt, typename TaskFunc, typename NFunc>
 void stencil(parallel_execution_tbb &p, InputIt first, InputIt last, OutputIt firstOut, TaskFunc && taskf, NFunc && neighbor ) {

    int numElements = last - first;
    int elemperthr = numElements/p.num_threads;
    tbb::task_group g;
 
    for(int i=1;i<p.num_threads;i++){
      


       g.run( [&neighbor, &taskf, first, firstOut, elemperthr, i, last, p ]() {
                auto begin = first + (elemperthr * i);
                auto end = first + (elemperthr * (i+1));
                if( i == p.num_threads-1) end = last;
                auto out = firstOut + (elemperthr * i);
                while(begin!=end){
                  auto neighbors = neighbor(begin);
                  *out = taskf(begin, neighbors);
                  begin++;
                  out++;
                }
              }
        );
    }
   //MAIN 
   auto end = first + elemperthr;
   while(first!=end){
      auto neighbors = neighbor(first);
      *firstOut = taskf(first, neighbors);
      first++;
      firstOut++;
   }

   g.wait();
 
}



template <typename InputIt, typename OutputIt, typename ... MoreIn, typename TaskFunc, typename NFunc>
 void stencil(parallel_execution_tbb &p, InputIt first, InputIt last, OutputIt firstOut, TaskFunc && taskf, NFunc && neighbor, MoreIn ... inputs ) {

     int numElements = last - first;
     int elemperthr = numElements/p.num_threads;
     tbb::task_group g;

     for(int i=1;i<p.num_threads;i++){

        
        g.run([&neighbor, &taskf, first, firstOut, elemperthr, i, last, p,inputs...]( )mutable{
        auto begin = first + (elemperthr * i);
        auto end = first + (elemperthr * (i+1));

               if(i==p.num_threads-1) end = last;

               auto out = firstOut + (elemperthr * i);
        
               GetStart(elemperthr, i, inputs ...);
               while(begin!=end){
                 auto neighbors = neighbor(begin);
                 *out = taskf(*begin, neighbors,inputs...);
                 begin++;
                 NextInputs( inputs ... );
                 out++;
               }
            });
    }

   //MAIN 
   auto end = first + elemperthr;
   while(first!=end){
      auto neighbors = neighbor(first);
      *firstOut = taskf(*first, neighbors, inputs...);
      first++;
      NextInputs( inputs ... );
      firstOut++;
   }


   //Join threads
   g.wait();
}

}
#endif

#endif