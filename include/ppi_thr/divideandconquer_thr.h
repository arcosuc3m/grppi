/**
* @version		GrPPI v0.1
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

#ifndef GRPPI_DIVIDEANDCONQUER_THR_H
#define GRPPI_DIVIDEANDCONQUER_THR_H

#include <thread>
#include <atomic>


using namespace std;
namespace grppi{
template <typename Input, typename Output, typename DivFunc, typename TaskFunc, typename MergeFunc>
inline void internal_divide_and_conquer(parallel_execution_thr &p, Input &problem, Output &output,
                                        DivFunc const &divide, TaskFunc const &task, MergeFunc const &merge,
                                        std::atomic<int>& num_threads) {

   if(num_threads.load()>0){
      auto subproblems = divide(problem);

      if(subproblems.size()>1){
        std::vector<Output> partials(subproblems.size());
//        get_num_threads() -= subproblems.size();
        int division = 1;
        std::vector<std::thread> tasks;
        auto i = subproblems.begin();
        for(i = subproblems.begin()+1; i != subproblems.end() && num_threads.load()>0 ; i++, division++){
            //THREAD
            tasks.push_back(
               std::thread(
                   [&](auto i,int division){
                    // Register the thread in the execution model
                    p.register_thread(); 

                    internal_divide_and_conquer(p, *i, partials[division], divide, task, merge, num_threads);

                    // Deregister the thread in the execution model
                    p.deregister_thread();
                  },
                   i, division
               )
            );

            num_threads--;
            //END TRHEAD
        }

        for(i; i != subproblems.end(); i++){
              divide_and_conquer(sequential_execution {},*i,partials[division], divide, task, merge);
        }
          //Main thread works on the first subproblem.

        internal_divide_and_conquer(p, *subproblems.begin(), partials[0], divide, task, merge, num_threads);
          //JOIN
        for(int i=0; i< tasks.size(); i++){
          tasks[i].join();
        }

        for(int i = 0; i<partials.size();i++){ // MarcoA - this is moved to the user code
           merge(partials[i], output);
        }
      }else{
        task(problem, output);
      }
    }else{
        divide_and_conquer(sequential_execution {}, problem, output, divide, task, merge);
    }
}






template <typename Input, typename Output, typename DivFunc, typename TaskFunc, typename MergeFunc>
inline void divide_and_conquer(parallel_execution_thr& p, Input & problem, Output & output,
            DivFunc const & divide, TaskFunc const & task, MergeFunc const & merge) {

    p.register_thread();
    std::atomic<int> num_threads ( p.get_num_threads() );
    
    if(num_threads.load()>0){
      auto subproblems = divide(problem);

      if(subproblems.size()>1){
        std::vector<Output> partials(subproblems.size());
	    int division = 1;
	    std::vector<std::thread> tasks;
        auto i = subproblems.begin();
        for(i = subproblems.begin()+1; i != subproblems.end(); i++, division++){
            //THREAD
	        tasks.push_back(
		       std::thread(
                   [&](auto i,int division){ 
                    // Register the thread in the execution model
                    p.register_thread();

                    internal_divide_and_conquer(p, *i, partials[division], divide, task, merge, num_threads);
                   
                    // Deregister the thread in the execution model
                    p.deregister_thread();
                  },
                   i, division )
	        );
            num_threads--;
            //END TRHEAD
        }
        for(i; i != subproblems.end(); i++){
              divide_and_conquer(sequential_execution {},*i,partials[division], divide, task, merge);
        }
          //Main thread works on the first subproblem.
        if(num_threads.load()>0){
            internal_divide_and_conquer(p, *subproblems.begin(), partials[0], divide, task, merge, num_threads);
        }else{
            divide_and_conquer(sequential_execution {}, *subproblems.begin(), partials[0], divide, task, merge);
        }

          //JOIN
        for(int i=0; i< tasks.size(); i++){
          tasks[i].join();
        }

        for(int i = 0; i<partials.size();i++){ // MarcoA - this is moved to the user code
           merge(partials[i], output);
           p.deregister_thread();
        }
       }else{
        task(problem, output);
      }
    }else{
        divide_and_conquer(sequential_execution {}, problem, output, divide, task, merge);
    }

}


/*

template <typename InputIt, typename OutputIt, typename ... MoreIn, typename TaskFunc>
inline void Reduce( InputIt first, InputIt last, OutputIt firstOut, TaskFunc const & taskf, MoreIn ... inputs ) {
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
