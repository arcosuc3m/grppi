/**
* @version    GrPPI v0.2
* @copyright    Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license    GNU/GPL, see LICENSE.txt
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

#ifndef GRPPI_POLY_STREAM_REDUCE_H
#define GRPPI_POLY_STREAM_REDUCE_H

#include "../common/support.h"
#include "polymorphic_execution.h"

namespace grppi{


template <typename Identity, typename Combiner, typename Consumer, typename Generator>
void stream_reduce_multi_impl(polymorphic_execution & ex, int window_size, int offset, 
                              Identity identity, Generator && generate_op, 
                              Combiner && combine_op, Consumer && consume_op)
{
}

template <typename E, typename ... O, typename Identity, 
          typename Generator, typename Combiner, typename Consumer,
          internal::requires_execution_not_supported<E> = 0>
void stream_reduce_multi_impl(polymorphic_execution & ex, int windowsize, int offset, 
                              Identity identity, Generator && gen, 
                              Combiner && comb, Consumer && cons) 
{
  stream_reduce_multi_impl<O...>(ex, 
    windowsize, offset, identity,
    std::forward<Generator>(gen), 
    std::forward<Combiner>(cons), 
    std::forward<Consumer>(cons)
  );
}

template <typename E, typename ... O, typename Identity, 
          typename Generator, typename Combiner, typename Consumer,
          internal::requires_execution_supported<E> = 0>
void stream_reduce_multi_impl(polymorphic_execution & ex, int windowsize, int offset,
                              Identity identity, Generator && gen, 
                              Combiner && comb, Consumer && cons) 
{
  if (typeid(E) == ex.type()) {
    stream_reduce(*ex.execution_ptr<E>(), 
      windowsize, offset, identity,
      std::forward<Generator>(gen),
      std::forward<Combiner>(comb), 
      std::forward<Consumer>(cons)
    );
  }
  else {
    stream_reduce_multi_impl<O...>(ex, 
      windowsize, offset, identity,
      std::forward<Generator>(gen),
      std::forward<Combiner>(comb), 
      std::forward<Consumer>(cons)
    );
  }
}

/**
\addtogroup stream_reduce_pattern
@{
*/

/**
\addtogroup stream_reduce_pattern_poly Polymorphic parallel stream reduce pattern
Polymorphic parallel implementation of the \ref md_stream-reduce pattern.
@{
*/

/**
\brief Invoke [stream reduce pattern](@ref md_stream-reduce) on a stream with
polymorphic parallel execution.
\tparam Identity Type of the identity value used by the combiner.
\tparam Generator Callable type used for generating data items.
\tparam Combiner Callable type used for data items combination.
\tparam Consumer Callable type used for consuming data items.
\param ex Polymorphic parallel execution policy object.
\param window_size Number of consecutive items to be reduced.
\param offset Number of items after of which a new reduction is started.
\param identity Identity value for the combination.
\param generate_op Generation operation.
\param combine_op Combination operation.
\param consume_op Consume operation.
*/
template <typename Identity, typename Generator, typename Combiner, typename Consumer>
void stream_reduce(polymorphic_execution & ex, 
      int windowsize, int offset, Identity identity,
      Generator && gen, Combiner && comb, Consumer && cons) 
{
  stream_reduce_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(ex, windowsize, offset, identity,
      std::forward<Generator>(gen),
      std::forward<Combiner>(comb), 
      std::forward<Consumer>(cons)
   );
}

/**
@}
@}
*/

} // end namespace grppi

#endif
