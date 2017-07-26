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

#ifndef GRPPI_OMP_STENCIL_H
#define GRPPI_OMP_STENCIL_H

#ifdef GRPPI_OMP
#include "parallel_execution_omp.h"


namespace grppi{
template <typename InputIt, typename OutputIt, typename Operation, typename NFunc>
 void stencil(parallel_execution_omp &p, InputIt first, InputIt last, OutputIt firstOut, Operation && op, NFunc && neighbor ) {

    int numElements = last - first;
    int elemperthr = numElements/p.concurrency_degree();
    #pragma omp parallel
    {
    #pragma omp single nowait
    { 
    for(int i=1;i<p.concurrency_degree();i++){
       #pragma omp task firstprivate(i)
       {
         auto begin = first + (elemperthr * i);
         auto end = first + (elemperthr * (i+1));
      
         if( i == p.concurrency_degree()-1) end = last;

         auto out = firstOut + (elemperthr * i);

         while(begin!=end){
           auto neighbors = neighbor(begin);
           *out = op(begin, neighbors);
           begin++;
           out++;
        }
      }
    }
   //MAIN
   auto begin = first; 
   auto end = first + elemperthr;
   auto out = firstOut;
   while(begin!=end){
      auto neighbors = neighbor(begin);
      *out = op(begin, neighbors);
      begin++;
      out++;
   }
   #pragma omp taskwait
   }
   }
}


template <typename InputIt, typename OutputIt, typename ... MoreIn, typename Operation, typename NFunc>
void internal_stencil(parallel_execution_omp & p, InputIt first, InputIt last, OutputIt firstOut, Operation && op, NFunc && neighbor, int i, int elemperthr, MoreIn ... inputs ){
   auto begin = first + (elemperthr * i);
   auto end = first + (elemperthr * (i+1));
   if(i==p.concurrency_degree()-1) end = last;

   auto out = firstOut + (elemperthr * i);

   advance_iterators((elemperthr*i), inputs ...);
   while(begin!=end){
      auto neighbors = neighbor(begin,inputs ... );
      *out = op(begin, neighbors);
      begin++;
      advance_iterators( inputs ... );
      out++;
   }
}


template <typename InputIt, typename OutputIt, typename ... MoreIn, typename Operation, typename NFunc>
void stencil(parallel_execution_omp & p, InputIt first, InputIt last, OutputIt firstOut, Operation && op, NFunc && neighbor, MoreIn ... inputs ) {

   int numElements = last - first;
   int elemperthr = numElements/p.concurrency_degree();
   #pragma omp parallel 
   {
   #pragma omp single nowait
   {

   for(int i=1;i<p.concurrency_degree();i++){
       #pragma omp task firstprivate(i)// firstprivate(inputs...)
       {
          internal_stencil(p,first,last,firstOut,std::forward<Operation>(op),std::forward<NFunc>(neighbor),i,elemperthr, inputs...);
       }
    }

   //MAIN
   auto begin = first;
   auto out = firstOut; 
   auto end = first + elemperthr;
   while(begin!=end){
      auto neighbors = neighbor(begin,inputs...);
      *out = op(*begin, neighbors);
      begin++;
      advance_iterators( inputs ... );
      out++;
   }

   #pragma omp taskwait
   }
   }


}


}
#endif




#endif
