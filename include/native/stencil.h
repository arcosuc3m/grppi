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

#ifndef GRPPI_STENCIL_THR_H
#define GRPPI_STENCIL_THR_H

namespace grppi{
template <typename InputIt, typename OutputIt, typename Operation, typename NFunc>
 void stencil(parallel_execution_native &p, InputIt first, InputIt last, OutputIt firstOut, Operation && op, NFunc && neighbor ) {

    std::vector<std::thread> tasks;
    int numElements = last - first;
    int elemperthr = numElements/p.num_threads;
 
    for(int i=1;i<p.num_threads;i++){
       auto begin = first + (elemperthr * i);
       auto end = first + (elemperthr * (i+1));
      
       if( i == p.num_threads-1) end = last;

       auto out = firstOut + (elemperthr * i);

       tasks.push_back(
           std::thread( [&](InputIt begin, InputIt end, OutputIt out){
              // Register the thread in the execution model
              p.register_thread();

              while(begin!=end){
                auto neighbors = neighbor(begin);
                *out = op(begin, neighbors);
                begin++;
                out++;
              }

              // Deregister the thread in the execution model
              p.deregister_thread();
           },
           begin, end, out)
      );
    }
   //MAIN 
   auto end = first + elemperthr;
   while(first!=end){
      auto neighbors = neighbor(first);
      *firstOut = op(first, neighbors);
      first++;
      firstOut++;
   }

   //Join threads
   for(int i=0;i<p.num_threads-1;i++){
      tasks[i].join();
   }
 
}

template <typename InputIt, typename OutputIt, typename ... MoreIn, typename Operation, typename NFunc>
 void stencil(parallel_execution_native &p, InputIt first, InputIt last, OutputIt firstOut, Operation && op, NFunc && neighbor, MoreIn ... inputs ) {

     std::vector<std::thread> tasks;
     int numElements = last - first;
     int elemperthr = numElements/p.num_threads;

     for(int i=1;i<p.num_threads;i++){

        auto begin = first + (elemperthr * i);
        auto end = first + (elemperthr * (i+1));

	if(i==p.num_threads-1) end = last;

        auto out = firstOut + (elemperthr * i);
        
        tasks.push_back(
            std::thread( [&](InputIt begin, InputIt end, OutputIt out, int i, int n, MoreIn ... inputs){

               // Register the thread in the execution model
               p.register_thread();
               
               advance_iterators(n*i, inputs ...);
               while(begin!=end){
                 auto neighbors = neighbor(begin);
                 *out = op(*begin, neighbors,inputs...);
                 begin++;
                 advance_iterators( inputs ... );
                 out++;
               }

               // Deregister the thread in the execution model
               p.deregister_thread();
            },
            begin, end, out, i, elemperthr,inputs ...)
       );
    }

   //MAIN 
   auto end = first + elemperthr;
   while(first!=end){
      auto neighbors = neighbor(first);
      *firstOut = op(*first, neighbors, inputs...);
      first++;
      advance_iterators( inputs ... );
      firstOut++;
   }


   //Join threads
   for(int i=0;i<p.num_threads-1;i++){
      tasks[i].join();
   }

}
}
#endif
