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

#ifndef GRPPI_POLY_REDUCE_H
#define GRPPI_POLY_REDUCE_H

#include "common/polymorphic_execution.h"
#include "common/support.h"

namespace grppi{

template <typename InputIt, typename Combiner>
typename std::result_of< Combiner(typename std::iterator_traits<InputIt>::value_type, typename std::iterator_traits<InputIt>::value_type) >::type
reduce_multi_impl(polymorphic_execution &e, InputIt first, InputIt last, Combiner &&combine_op)
{
}



template <typename E, typename ... O,
          typename InputIt, typename Combiner,
          internal::requires_execution_not_supported<E> = 0>
typename std::result_of< Combiner(typename std::iterator_traits<InputIt>::value_type, typename std::iterator_traits<InputIt>::value_type) >::type
reduce_multi_impl(polymorphic_execution &e, InputIt first, InputIt last, Combiner &&combine_op)
{
  reduce_multi_impl<O...>(e, first, last, std::forward<Combiner>(combine_op));
}



template <typename E, typename ... O,
          typename InputIt, typename Combiner,
          internal::requires_execution_supported<E> = 0>
typename std::result_of< Combiner(typename std::iterator_traits<InputIt>::value_type, typename std::iterator_traits<InputIt>::value_type) >::type
reduce_multi_impl(polymorphic_execution &e, InputIt first, InputIt last, Combiner &&combine_op)
{
  if (typeid(E) == e.type()) {
    reduce(*e.execution_ptr<E>(), 
        first, last, std::forward<Combiner>(combine_op));
  }
  else {
    reduce_multi_impl<O...>(e, first, last, std::forward<Combiner>(combine_op));
  }
}



/// Runs a reduce pattern with an input sequence, and a combination operation
/// InputIt: Iterator for the input sequence.
/// Combiner: Combiner functor type
template <typename InputIt, typename Combiner>
typename std::result_of< Combiner(typename std::iterator_traits<InputIt>::value_type, typename std::iterator_traits<InputIt>::value_type) >::type
 reduce(polymorphic_execution & e, InputIt first, InputIt last, Combiner &&combine_op)
{
  reduce_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb
  >(e, first, last, std::forward<Combiner>(combine_op));
}

} // end namespace grppi

#endif
