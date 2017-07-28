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

#ifndef GRPPI_NATIVE_DIVIDECONQUER_H
#define GRPPI_NATIVE_DIVIDECONQUER_H

#include "parallel_execution_native.h"

#include <thread>
#include <atomic>

namespace grppi {

template <typename Input, typename Divider, typename Solver, typename Combiner>
typename std::result_of<Solver(Input)>::type 
internal_divide_conquer(parallel_execution_native &p, 
                            Input & input,
                            Divider && divide_op, Solver && solve_op, 
                            Combiner && combine_op,
                            std::atomic<int> & num_threads) 
{
  // Sequential execution fo internal implementation
  using Output = typename std::result_of<Solver(Input)>::type;
  sequential_execution seq;
  Output out;
   if(num_threads.load()>0){
      auto subproblems = divide_op(input);

      if(subproblems.size()>1){
        std::vector<Output> partials(subproblems.size()-1);
//        num_threads -= subproblems.size();
        int division = 0;
        std::vector<std::thread> tasks;
        auto i = subproblems.begin();
        for(i = subproblems.begin()+1; i != subproblems.end() && num_threads.load()>0 ; i++, division++){
            //THREAD
          tasks.emplace_back([&](auto i, int division) {
            auto manager = p.thread_manager();

            partials[division] = internal_divide_conquer(p, *i,
              std::forward<Divider>(divide_op),
              std::forward<Solver>(solve_op), 
              std::forward<Combiner>(combine_op), 
              num_threads);

          }, i, division);

          num_threads--;
          //END TRHEAD
        }

        for(i; i != subproblems.end(); i++){
              partials[division] = divide_conquer(seq,*i, 
                  std::forward<Divider>(divide_op), 
                  std::forward<Solver>(solve_op), 
                  std::forward<Combiner>(combine_op));
        }
          //Main thread works on the first subproblem.

        out = internal_divide_conquer(p, *subproblems.begin(), 
            std::forward<Divider>(divide_op), 
            std::forward<Solver>(solve_op), 
            std::forward<Combiner>(combine_op), num_threads);
          //JOIN
        for(int i=0; i< tasks.size(); i++){
          tasks[i].join();
        }

        for(int i = 0; i<partials.size();i++){ // MarcoA - this is moved to the user code
           out =combine_op(out,partials[i]);
        }
      }else{
        out = solve_op(input);
      }
    }else{
        return divide_conquer(seq, input, 
            std::forward<Divider>(divide_op), 
            std::forward<Solver>(solve_op), 
            std::forward<Combiner>(combine_op));
    }
    return out;
}

/**
\addtogroup divide_conquer_pattern
@{
\addtogroup divide_conquer_pattern_native Native parallel divide/conquer pattern
\brief Native parallel implementation of the \ref md_divide-conquer.
@{
*/

/**
\brief Invoke \ref md_divide-conquer with native
parallel execution.
\tparam Input Type used for the input problem.
\tparam Divider Callable type for the divider operation.
\tparam Solver Callable type for the solver operation.
\tparam Combiner Callable type for the combiner operation.
\param ex Sequential execution policy object.
\param input Input problem to be solved.
\param divider_op Divider operation.
\param solver_op Solver operation.
\param combiner_op Combiner operation.
*/
template <typename Input, typename Divider, typename Solver, typename Combiner>
typename std::result_of<Solver(Input)>::type 
divide_conquer(parallel_execution_native & ex, 
                   Input & problem,
                   Divider && divide_op, Solver && solve_op, 
                   Combiner && combine_op) 
{
  // Sequential execution fo internal implementation
  sequential_execution seq;
  std::atomic<int> num_threads{ex.concurrency_degree()-1};
    
  if (num_threads.load()>0) {
    return divide_conquer(seq, problem, 
        std::forward<Divider>(divide_op), 
        std::forward<Solver>(solve_op), 
        std::forward<Combiner>(combine_op));
  }

  auto subproblems = divide_op(problem);

  if(subproblems.size()>1) {
    using Output = typename std::result_of<Solver(Input)>::type;
    std::vector<Output> partials(subproblems.size()-1);
    int division = 0;
    std::vector<std::thread> tasks;
    auto i = subproblems.begin();
    for(i = subproblems.begin()+1; i != subproblems.end() && num_threads.load()>0; i++, division++) {
      //THREAD
      tasks.emplace_back([&](auto i, int division) { 
        auto manager = ex.thread_manager();
        partials[division] = internal_divide_conquer(ex, *i, 
          std::forward<Divider>(divide_op), 
          std::forward<Solver>(solve_op), 
          std::forward<Combiner>(combine_op), 
          num_threads);
      }, i, division);

      num_threads--;
      //END TRHEAD
    }

    for(i; i != subproblems.end(); i++) {
      partials[division] = divide_conquer(seq, *i, 
          std::forward<Divider>(divide_op), 
          std::forward<Solver>(solve_op), 
          std::forward<Combiner>(combine_op));
    }
        
    //Main thread works on the first subproblem.
    Output out = internal_divide_conquer(ex, *subproblems.begin(), 
        std::forward<Divider>(divide_op), 
        std::forward<Solver>(solve_op), 
        std::forward<Combiner>(combine_op), 
        num_threads);
       

    //JOIN
    for (auto && t : tasks) { t.join(); }

    for (auto && p : partials) { out = combine_op(out,p); }

    return out;
  }
  else {
    return solve_op(problem);
  }
}

/**
@}
@}
*/

}

#endif
