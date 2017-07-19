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

#ifndef GRPPI_POLY_STREAM_FILTER_H
#define GRPPI_POLY_STREAM_FILTER_H

#include "common/support.h"
#include "polymorphic_execution.h"

namespace grppi{

template <typename Generator, typename Predicate, typename Consumer>
void stream_filter_multi_impl(polymorphic_execution, Generator && generate_op,  
                              Predicate predicate_op, Consumer consume_op)
{
}

template <typename E, typename ... O,
          typename Generator, typename Predicate, typename Consumer,
          internal::requires_execution_not_supported<E> = 0>
void stream_filter_multi_impl(polymorphic_execution & ex, 
                              Generator && generate_op, 
                              Predicate && predicate_op, Consumer && consume_op) 
{
  stream_filter_multi_impl<O...>(ex, std::forward<Generator>(generate_op),
    std::forward<Predicate>(predicate_op), std::forward<Consumer>(consume_op));
}



template <typename E, typename ... O,
          typename Generator, typename Predicate, typename Consumer,
          internal::requires_execution_supported<E> = 0>
void stream_filter_multi_impl(polymorphic_execution & ex, 
                              Generator && generate_op, 
                              Predicate && predicate_op, 
                              Consumer && consume_op) 
{
  if (typeid(E) == ex.type()) {
    stream_filter(*ex.execution_ptr<E>(), 
      std::forward<Generator>(generate_op), std::forward<Predicate>(predicate_op), 
      std::forward<Consumer>(consume_op));
  }
  else {
    stream_filter_multi_impl<O...>(ex, std::forward<Generator>(generate_op),
        std::forward<Predicate>(predicate_op), std::forward<Consumer>(consume_op));
  }
}


/** 
\addtogroup filter_pattern
@{
*/

/**
\addtogroup filter_poly Polymorphic parallel filter pattern.
\brief Polymorphic parallel implementation fo the \ref md_stream-filter pattern.
@{
*/

/**
\brief Invoke [stream filter pattern](@ref md_stream-filter pattern) on a data
sequence with polymorphic execution policy.
\tparam Generator Callable type for value generator.
\tparam Predicate Callable type for filter predicate.
\tparam Consumer Callable type for value consumer.
\param ex Polymorphic execution policy object.
\param generate_op Generator callable object.
\param predicate_op Predicate callable object.
\param consume_op Consumer callable object.
*/
template <typename Generator, typename Predicate, typename Consumer>
void stream_filter(polymorphic_execution & ex, Generator && generate_op, 
                   Predicate && predicate_op, Consumer && consume_op) 
{
  stream_filter_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(ex, std::forward<Generator>(generate_op),
      std::forward<Predicate>(predicate_op),
      std::forward<Consumer>(consume_op));
}

/**
@}
@}
*/


} // end namespace grppi

#endif
