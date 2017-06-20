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

#ifndef GRPPI_FARM_OMP_H
#define GRPPI_FARM_OMP_H

#ifdef GRPPI_OMP

namespace grppi
{

template <typename GenFunc, typename Operation>
void farm(parallel_execution_omp &p, GenFunc &&in, Operation &&op) {
	
    Queue<typename std::result_of<GenFunc()>::type> queue(DEFAULT_SIZE, p.lockfree);
    #pragma omp parallel
    {
	#pragma omp single nowait
	{
            //Create threads
            for( int i = 0; i < p.num_threads; i++ ) {
                #pragma omp task shared(queue)
		{
                    typename std::result_of<GenFunc()>::type item;
                    item = queue.pop() ;
                    while( item ) {
                        op( item.value() );
                        item = queue.pop() ;
                    }
               	}
            }
		
            //Generate elements
            while( 1 ) {
                auto k = in();
                queue.push( k ) ;
                if( !k ) {
                    for( int i = 1; i < p.num_threads; i++ )
                        queue.push(k);
                    break;
                }
            }
            //Join threads
            #pragma omp taskwait
        }
    }	
}

template <typename Operation>
farm_info<parallel_execution_omp,Operation> farm(parallel_execution_omp &p, Operation && op){
   return farm_info<parallel_execution_omp, Operation>(p,op);
}
}
#endif

#endif