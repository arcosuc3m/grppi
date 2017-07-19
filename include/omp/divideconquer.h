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

#ifndef GRPPI_OMP_DIVIDECONQUER_H
#define GRPPI_OMP_DIVIDECONQUER_H

#ifdef GRPPI_OMP

#include <thread>

#include "parallel_execution_omp.h"

namespace grppi{

template <typename Input, typename Divider, typename Solver, typename Combiner>
typename std::result_of<Solver(Input)>::type 
internal_divide_conquer(parallel_execution_omp & ex, 
                            Input & input, 
                            Divider && divide_op, Solver && solve_op, 
                            Combiner && combine_op, 
                            std::atomic<int> & num_threads) 
{
  // Sequential execution fo internal implementation
  using  Output = typename std::result_of<Solver(Input)>::type;
  sequential_execution seq;
  Output out;
  if(num_threads.load()>0) {
    auto subproblems = divide_op(input);


    if(subproblems.size()>1) {
      std::vector<Output> partials(subproblems.size()-1);
      int division = 0;
      auto i = subproblems.begin()+1;
      for(i; i!=subproblems.end() && num_threads.load()>0; i++, division++){
        //THREAD
        #pragma omp task firstprivate(i, division) shared(divide_op, solve_op, \
                combine_op, partials, num_threads)
        {
          partials[division] = internal_divide_conquer(ex, *i, 
              std::forward<Divider>(divide_op), 
              std::forward<Solver>(solve_op), 
              std::forward<Combiner>(combine_op), num_threads);
         //END TRHEAD
         }
         num_threads --;
             
      }
      for(i; i != subproblems.end(); i++){
        partials[division] = divide_conquer(seq, *i, 
            std::forward<Divider>(divide_op), 
            std::forward<Solver>(solve_op), 
            std::forward<Combiner>(combine_op));
      }
          
      //Main thread works on the first subproblem.
      out = internal_divide_conquer(ex, *subproblems.begin(), 
          std::forward<Divider>(divide_op), 
          std::forward<Solver>(solve_op), 
          std::forward<Combiner>(combine_op), num_threads);

      #pragma omp taskwait

      for(int i = 0; i<partials.size();i++){ 
        out = combine_op(out,partials[i]);
      }
    }
    else {
      out = solve_op(input);
    }
  }
  else {
    return divide_conquer(seq, input, std::forward<Divider>(divide_op), 
        std::forward<Solver>(solve_op), std::forward<Combiner>(combine_op));
  }
  return out;
}

/**
\addtogroup divide_conquer_pattern
@{
*/

/**
\addtogroup divide_conquer_pattern_omp OpenMP parallel divide/conquer pattern.
\brief OpenMP parallel implementation of the \ref md_divide-conquer pattern.
@{
*/

/**
\brief Invoke [divide/conquer pattern](@ref md_divide-conquer) with OpenMP
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
divide_conquer(parallel_execution_omp & ex, 
                   Input & input, 
                   Divider && divide_op, Solver && solve_op, 
                   Combiner && combine_op) 
{
  std::atomic<int> num_threads{ex.num_threads-1};

  sequential_execution seq;

  using Output = typename std::result_of<Solver(Input)>::type;

  if (num_threads.load()<=0) {
    return divide_conquer(seq, input, 
        std::forward<Divider>(divide_op), 
        std::forward<Solver>(solve_op), 
        std::forward<Combiner>(combine_op));
  }

  auto subproblems = divide_op(input);
  if (subproblems.size()<=1) {
    return solve_op(input);
  }

  std::vector<Output> partials(subproblems.size()-1);
  int division = 0;
  
  Output out;

  #pragma omp parallel
  {
    #pragma omp single nowait
    { 
      auto i = subproblems.begin() + 1;
      for (i; i!=subproblems.end() && num_threads.load()>0; i++, division++) {
        //THREAD
        #pragma omp task firstprivate(i,division) \
                shared(partials,divide_op,solve_op,combine_op,num_threads)
        {
          partials[division] = internal_divide_conquer(ex, *i,
              std::forward<Divider>(divide_op), 
              std::forward<Solver>(solve_op), 
              std::forward<Combiner>(combine_op), num_threads);
          //END TRHEAD
        }
        num_threads --;
      }

      for (i; i!=subproblems.end(); i++) {
        partials[division] = divide_conquer(seq, *i, 
            std::forward<Divider>(divide_op), 
            std::forward<Solver>(solve_op), 
            std::forward<Combiner>(combine_op));
      }

      //Main thread works on the first subproblem.
      if (num_threads.load()>0) {
        out = internal_divide_conquer(ex, *subproblems.begin(),
            std::forward<Divider>(divide_op), 
            std::forward<Solver>(solve_op), 
            std::forward<Combiner>(combine_op), num_threads);
      }
      else {
        out = divide_conquer(seq, *subproblems.begin(),
            std::forward<Divider>(divide_op), 
            std::forward<Solver>(solve_op), 
            std::forward<Combiner>(combine_op));
      }
      #pragma omp taskwait
    }
  }

  for (auto && p : partials) { out = combine_op(out,p); }
  return out;
}

/**
@}
@}
*/

}

#endif

#endif
