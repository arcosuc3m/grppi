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

#ifndef GRPPI_OMP_STREAM_FILTER_H
#define GRPPI_OMP_STREAM_FILTER_H

#ifdef GRPPI_OMP

#include "parallel_execution_omp.h"

namespace grppi{
template <typename Generator, typename Predicate, typename Consumer>
 void stream_filter(parallel_execution_omp &p, Generator && gen, Predicate && pred, Consumer && cons ) {


    mpmc_queue< typename std::result_of<Generator()>::type > queue(p.queue_size, p.lockfree);
    mpmc_queue< typename std::result_of<Generator()>::type > outqueue(p.queue_size, p.lockfree);
    #pragma omp parallel
    {
    #pragma omp single nowait 
    {

    //THREAD 1-(N-1) EXECUTE FILTER AND PUSH THE VALUE IF TRUE
    for(int i=0; i< p.num_threads - 1; i++){

        #pragma omp task shared(queue, outqueue)
        {
            typename std::result_of<Generator()>::type item;
            item = queue.pop();
            while( item ){
                 if(pred(item.value()))
                      outqueue.push(item);
                 item = queue.pop();
            }
            outqueue.push(item);
         }
     }
     #pragma omp task shared(outqueue)
     {
         //LAST THREAD CALL FUNCTION OUT WITH THE FILTERED ELEMENTS

         int nend = 0;
         typename std::result_of<Generator()>::type item;
         item = outqueue.pop();
         while(nend != p.num_threads - 1){
            if(!item){
                nend++;
                if(nend == p.num_threads - 1) break;
            }
            else {
                cons(item.value());
            }
            item = outqueue.pop();
         }
     }

    //THREAD 0 ENQUEUE ELEMENTS
    while(1){
        auto k = gen();
        queue.push(k);
        if( !k ){
           for(int i = 0; i< p.num_threads-1; i++){
              queue.push(k);
           }
           break;
        }
    }
    #pragma omp taskwait
    }
    }

}

template <typename Predicate>
filter_info<parallel_execution_omp, Predicate> stream_filter(parallel_execution_omp &p, Predicate && pred){
   return filter_info<parallel_execution_omp, Predicate>(p, std::forward<Predicate>(pred) );

}
}
#endif

#endif
