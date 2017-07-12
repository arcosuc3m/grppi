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

#ifndef GRPPI_SEQ_MAPREDUCE_H
#define GRPPI_SEQ_MAPREDUCE_H

#include "reduce.h"

namespace grppi{

/**
\addtogroup mapreduce_pattern
@{
*/

/**
\addtogroup mapreduce_pattern_seq Sequential map/reduce pattern
Sequential implementation of the \ref md_map-reduce pattern.
@{
*/

/**
\brief Invoke [map/reduce pattern](\ref md_map-reduce) on a data sequence with 
sequential execution.
\tparam InputIt Iterator type used for input sequence.
\tparam Result Result type of the reduction.
\tparam Transformer Callable type for the transformation operation.
\tparam Combiner Callable type for the combination operation of the reduction.
\param ex Sequential execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param identity Identity value for the combination operation.
\param transf_op Transformation operation.
\param combine_op Combination operation.
\return Result of the map/reduce operation.
*/
template <typename InputIt, typename Result, typename Transformer, 
          typename Combiner>
Result map_reduce(sequential_execution &, 
                  InputIt first, InputIt last, 
                  Result identity, 
                  Transformer &&  transform_op, Combiner && combine_op)
{
  Result out = identity;

  using namespace std;
  cerr << "---Initial: " << out << endl;

  while (first!=last) {
    auto x = transform_op(*first);
    out = combine_op(out,x);
    cerr << "---combining with " << x << endl;
    first++;
  }

  return out;
}

}
#endif
