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

#ifndef GRPPI_NATIVE_MAPREDUCE_H
#define GRPPI_NATIVE_MAPREDUCE_H

#include "../reduce.h"

namespace grppi{

/**
\addtogroup mapreduce_pattern
@{
*/

/**
\addtogroup mapreduce_pattern_native Native parallel map/reduce pattern
\brief Native parallel implementation of the \ref md_map-reduce pattern.
@{
*/

/**
\brief Invoke [map/reduce pattern](\ref md_map-reduce) on a data sequence with 
native parallel execution.
\tparam InputIt Iterator type used for input sequence.
\tparam Result Result type of the reduction.
\tparam Transformer Callable type for the transformation operation.
\tparam Combiner Callable type for the combination operation of the reduction.
\param ex Native parallel execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param identity Result value for the combination operation.
\param transf_op Transformation operation.
\param combine_op Combination operation.
\return Result of the map/reduce operation.
*/
template <typename InputIt, typename Result, typename Transformer, 
          typename Combiner>
Result map_reduce(parallel_execution_native & ex, 
                  InputIt first, InputIt last, Result identity, 
                  Transformer && transform_op,  Combiner &&combine_op)
{
  using namespace std;

  vector<Result> partial_results(ex.num_threads);

  const int num_elements = last - first;
  const int elements_per_thread = num_elements/ex.num_threads;
  sequential_execution seq{};

  vector<thread> tasks;
  for(int i=1;i<ex.num_threads;i++){    
    const auto begin = first + (elements_per_thread * i);
    const auto end = (i==ex.num_threads-1) ? 
        last : 
        (first + elements_per_thread * (i+1));

    tasks.emplace_back([&,begin,end,i](){
        ex.register_thread();
        partial_results[i] = map_reduce(seq, begin, end, partial_results[i], 
            forward<Transformer>(transform_op), forward<Combiner>(combine_op));
        ex.deregister_thread();
    });
  }

    partial_results[0] = map_reduce(seq, 
        first,( first+elements_per_thread ), partial_results[0], 
        forward<Transformer>(transform_op), 
        forward<Combiner>(combine_op));

    for (auto && t : tasks) { t.join(); }

    Result result = identity;
    for (auto && p : partial_results) { result = combine_op(result, p); } 

    return result;
}

/**
@}
@}
*/

}
#endif
