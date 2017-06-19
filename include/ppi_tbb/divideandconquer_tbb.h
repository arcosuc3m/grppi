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

#ifndef GRPPI_DIVIDEANDCONQUER_TBB_H
#define GRPPI_DIVIDEANDCONQUER_TBB_H

#ifdef GRPPI_TBB

#include <tbb/tbb.h>
namespace grppi{
template <typename Input, typename Output, typename DivFunc, typename TaskFunc, typename MergeFunc>
 void internal_divide_and_conquer(parallel_execution_tbb &p, Input & problem, Output & output,
            DivFunc && divide, TaskFunc && task, MergeFunc && merge, std::atomic<int>& num_threads) {
   
    // Sequential execution fo internal implementation
    sequential_execution seq;

    if(num_threads.load()>0){
       auto subproblems = divide(problem);


    if(subproblems.size()>1){
         std::vector<Output> partials(subproblems.size());
         int division = 1;
         tbb::task_group g;
         auto i = subproblems.begin()+1;
         for(i ; i != subproblems.end() && num_threads.load() > 0; i++, division++){
            //THREAD
            g.run(
              [&p, i, &partials, division, &divide, &task, &merge, &num_threads](){
                 internal_divide_and_conquer(p, *i, partials[division], std::forward<DivFunc>(divide), std::forward<TaskFunc>(task), std::forward<MergeFunc>(merge), num_threads);
              }
            );
              //END TRHEAD
            num_threads--;
          }
          //Main thread works on the first subproblem.
          for(i; i != subproblems.end(); i++){
              divide_and_conquer(seq,*i,partials[division], std::forward<DivFunc>(divide), std::forward<TaskFunc>(task), std::forward<MergeFunc>(merge));
          }

          internal_divide_and_conquer(p, *subproblems.begin(), partials[0], std::forward<DivFunc>(divide), std::forward<TaskFunc>(task), std::forward<MergeFunc>(merge), num_threads);

          g.wait();

          for(int i = 0; i<partials.size();i++){ // MarcoA - this is moved to the user code
              merge(partials[i], output);
          }

        }else{
          task(problem, output);
        }

     }else{
        divide_and_conquer(seq, problem, output, std::forward<DivFunc>(divide), std::forward<TaskFunc>(task), std::forward<MergeFunc>(merge));
     }

}

template <typename Input, typename Output, typename DivFunc, typename TaskFunc, typename MergeFunc>
 void divide_and_conquer(parallel_execution_tbb &p, Input & problem, Output & output,
            DivFunc && divide, TaskFunc && task, MergeFunc && merge) {

    // Sequential execution fo internal implementation
    sequential_execution seq;

    std::atomic<int> num_threads( p.num_threads );

    if(num_threads.load()>0){
       auto subproblems = divide(problem);

      if(subproblems.size()>1){
          std::vector<Output> partials(subproblems.size());
    	  int division = 1;
          tbb::task_group g;
             
          auto i = subproblems.begin()+1;
          for(i ; i != subproblems.end() && num_threads.load() > 0; i++, division++){
              //THREAD
              g.run(
                 [&p, i, &partials, division, &divide, &task, &merge, &num_threads](){
                     internal_divide_and_conquer(p, *i, partials[division], std::forward<DivFunc>(divide), std::forward<TaskFunc>(task), std::forward<MergeFunc>(merge), num_threads);
                  }
              );
              num_threads--;
              //END TRHEAD
          }
          for(i; i != subproblems.end(); i++){
              divide_and_conquer(seq,*i,partials[division], std::forward<DivFunc>(divide), std::forward<TaskFunc>(task), std::forward<MergeFunc>(merge));
          }
          //Main thread works on the first subproblem.

          if(num_threads.load()>0){
            internal_divide_and_conquer(p, *subproblems.begin(), partials[0], std::forward<DivFunc>(divide), std::forward<TaskFunc>(task), std::forward<MergeFunc>(merge), num_threads);
          }else{
            divide_and_conquer(seq, *subproblems.begin(), partials[0], std::forward<DivFunc>(divide), std::forward<TaskFunc>(task), std::forward<MergeFunc>(merge));
          }

          g.wait();

          for(int i = 0; i<partials.size();i++){ // MarcoA - this is moved to the user code
              merge(partials[i], output);
          }
        }else{
          task(problem, output);
        }
    }else{
       divide_and_conquer(seq, problem, output, std::forward<DivFunc>(divide), std::forward<TaskFunc>(task), std::forward<MergeFunc>(merge));
    }
}


/*

template <typename InputIt, typename OutputIt, typename ... MoreIn, typename TaskFunc>
 void Reduce( InputIt first, InputIt last, OutputIt firstOut, TaskFunc && taskf, MoreIn ... inputs ) {
    while( first != last ) {
        *firstOut = taskf( *first, *inputs ... );
        NextInputs( inputs... );
        first++;
        firstOut++;
    }
}
*/
}
#endif

#endif