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

#ifndef GRPPI_OMP_MAPREDUCE_H
#define GRPPI_OMP_MAPREDUCE_H

#ifdef GRPPI_OMP

#include "parallel_execution_omp.h"

namespace grppi {

/**
\addtogroup mapreduce_pattern
@{
\addtogroup mapreduce_pattern_omp OpenMP parallel map/reduce pattern
\brief OpenMP parallel implementation of the \ref md_map-reduce.
@{
*/

/**
\brief Invoke \ref md_map-reduce on a data sequence with 
native parallel execution.
\tparam InputIt Iterator type used for input sequence.
\tparam Result Result type of the reduction.
\tparam Transformer Callable type for the transformation operation.
\tparam Combiner Callable type for the combination operation of the reduction.
\param ex OpenMP parallel execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param identity Result value for the combination operation.
\param transf_op Transformation operation.
\param combine_op Combination operation.
\return Result of the map/reduce operation.
*/
template <typename InputIt, typename Transformer, typename Result, 
          typename Combiner>
Result map_reduce(parallel_execution_omp & ex, 
                    InputIt first, InputIt last, Result identity, 
                    Transformer &&  transform_op,  
                    Combiner && combine_op)
{
  using namespace std;
  Result result{identity};

  std::vector<Result> partial_results(ex.concurrency_degree());
  #pragma omp parallel
  {
    #pragma omp single nowait
    {
      int num_elements = distance(first,last);
      int elements_per_thread = num_elements/ex.concurrency_degree();
      sequential_execution seq{};

      for (int i=1;i<ex.concurrency_degree();i++) {    
        #pragma omp task firstprivate(i)
        {
          auto begin = next(first, elements_per_thread * i);
          auto end = (i==ex.concurrency_degree()-1) ? last :
              next(first, elements_per_thread * (i+1));
          partial_results[i] = map_reduce(seq, 
              begin, end, partial_results[i], 
              std::forward<Transformer>(transform_op), 
              std::forward<Combiner>(combine_op));
        }
      }

      partial_results[0] = map_reduce(seq, 
          first, first+elements_per_thread, partial_results[0], 
          std::forward<Transformer>(transform_op), 
          std::forward<Combiner>(combine_op));
      #pragma omp taskwait
    }
  }

  for (auto && p : partial_results){
    result = combine_op(result, p);
  } 
  return result;
}

/**
@}
@}
*/

}
#endif

#endif
