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

#ifndef GRPPI_DIVIDEANDCONQUER_OMP_H
#define GRPPI_DIVIDEANDCONQUER_OMP_H

#ifdef GRPPI_OMP

#include <thread>

namespace grppi{

template <typename Input, typename Output, typename DivFunc, typename Operation, typename MergeFunc>
Output internal_divide_and_conquer(parallel_execution_omp &p, Input & problem, Output init,
            DivFunc && divide, Operation && op, MergeFunc && merge, std::atomic<int> & num_threads) {
   
    // Sequential execution fo internal implementation
    sequential_execution seq;
    auto out = init;
    if(num_threads.load()>0){
       auto subproblems = divide(problem);


    if(subproblems.size()>1){
         std::vector<Output> partials(subproblems.size());
         int division = 1;
         auto i = subproblems.begin()+1;
         for(i; i != subproblems.end()&& num_threads.load()>0; i++, division++){
            //THREAD
             #pragma omp task firstprivate(i, division) shared(divide, op, merge, partials, num_threads)
             {
                 partials[division] = internal_divide_and_conquer(p, *i, partials[division], std::forward<DivFunc>(divide), std::forward<Operation>(op), std::forward<MergeFunc>(merge), num_threads);
              //END TRHEAD
             }
             num_threads --;
             
          }
          for(i; i != subproblems.end(); i++){
              partials[division] = divide_and_conquer(seq,*i,partials[division], divide, op, merge);
          }
          
          //Main thread works on the first subproblem.
          partials[0] = internal_divide_and_conquer(p, *subproblems.begin(), partials[0], std::forward<DivFunc>(divide), std::forward<Operation>(op), std::forward<MergeFunc>(merge), num_threads);

          #pragma omp taskwait

          for(int i = 0; i<partials.size();i++){ // MarcoA - this is moved to the user code
              merge(partials[i], out);
          }
        }else{
          out = op(problem);
        }
     }else{
        return divide_and_conquer(seq, problem, out, std::forward<DivFunc>(divide), std::forward<Operation>(op), std::forward<MergeFunc>(merge));
     }
     return out;



}

template <typename Input, typename Output, typename DivFunc, typename Operation, typename MergeFunc>
Output divide_and_conquer(parallel_execution_omp &p, Input & problem, Output init,
            DivFunc && divide, Operation && op, MergeFunc && merge) {

    // Sequential execution fo internal implementation
    sequential_execution seq;

    std::atomic<int> num_threads( p.num_threads );
    auto out = init;
    if(num_threads.load()>0){
       auto subproblems = divide(problem);

      if(subproblems.size()>1){
          std::vector<Output> partials(subproblems.size());
    	  int division = 1;
  
          #pragma omp parallel
          {
          #pragma omp single nowait
          { 
          auto i = subproblems.begin()+1;
          for(i; i != subproblems.end()&& num_threads.load()>0; i++, division++){
              //THREAD
              #pragma omp task firstprivate(i,division) shared(partials,divide,op,merge, num_threads)
              {
                  partials[division] = internal_divide_and_conquer(p, *i, partials[division], std::forward<DivFunc>(divide), std::forward<Operation>(op), std::forward<MergeFunc>(merge), num_threads);
              //END TRHEAD
              }
                  num_threads --;

          }
          for(i; i != subproblems.end(); i++){
              partials[division] = divide_and_conquer(seq,*i,partials[division], std::forward<DivFunc>(divide), std::forward<Operation>(op), std::forward<MergeFunc>(merge));
          }

    	  //Main thread works on the first subproblem.
        
         if(num_threads.load()>0){
            partials[0] = internal_divide_and_conquer(p, *subproblems.begin(), partials[0], std::forward<DivFunc>(divide), std::forward<Operation>(op), std::forward<MergeFunc>(merge), num_threads);
        }else{
            partials[0] = divide_and_conquer(seq, *subproblems.begin(), partials[0], std::forward<DivFunc>(divide), std::forward<Operation>(op), std::forward<MergeFunc>(merge));
        }
          #pragma omp taskwait
          }
          }
          for(int i = 0; i<partials.size();i++){ // MarcoA - this is moved to the user code
              merge(partials[i], out);
          }
        }else{
          out = op(problem);
        }
    }else{
        return divide_and_conquer(seq, problem, out, std::forward<DivFunc>(divide), std::forward<Operation>(op), std::forward<MergeFunc>(merge));
    }
    return out;
}


/*

template <typename InputIt, typename OutputIt, typename ... MoreIn, typename Operation>
 void Reduce( InputIt first, InputIt last, OutputIt firstOut, Operation && op, MoreIn ... inputs ) {
    while( first != last ) {
        *firstOut = op( *first, *inputs ... );
        NextInputs( inputs... );
        first++;
        firstOut++;
    }
}
*/
}
#endif

#endif
