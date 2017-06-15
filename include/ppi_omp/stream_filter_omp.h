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

#ifndef GRPPI_STREAM_FILTER_OMP_H
#define GRPPI_STREAM_FILTER_OMP_H

using namespace std;
namespace grppi{
template <typename GenFunc, typename FilterFunc, typename OutFunc>
 void stream_filter(parallel_execution_omp p, GenFunc const & in, FilterFunc const & filter, OutFunc const & out ) {

    Queue< typename std::result_of<GenFunc()>::type > queue(DEFAULT_SIZE, p.lockfree);
    Queue< typename std::result_of<GenFunc()>::type > outqueue(DEFAULT_SIZE, p.lockfree);
    #pragma omp parallel
    {
    #pragma omp single nowait 
    {

    //THREAD 1-(N-1) EXECUTE FILTER AND PUSH THE VALUE IF TRUE
    for(int i=0; i< p.num_threads - 1; i++){

        #pragma omp task shared(queue, outqueue)
        {
            typename std::result_of<GenFunc()>::type item;
            item = queue.pop();
            while( item ){
                 if(filter(item.value()))
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
         typename std::result_of<GenFunc()>::type item;
         item = outqueue.pop();
         while(nend != p.num_threads - 1){
            if(!item){
                nend++;
                if(nend == p.num_threads - 1) break;
            }
            else {
                out(item.value());
            }
            item = outqueue.pop();
         }
     }

    //THREAD 0 ENQUEUE ELEMENTS
    while(1){
        auto k = in();
        queue.push(k);
        if( k.end ){
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

template <typename FilterFunc>
FilterObj<parallel_execution_omp, FilterFunc> stream_filter(parallel_execution_omp p, FilterFunc && taskf){
   return FilterObj<parallel_execution_omp, FilterFunc>(p, taskf);

}
}
#endif
