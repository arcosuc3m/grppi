
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

#ifndef GRPPI_POLY_FARM_H
#define GRPPI_POLY_FARM_H

#include "polymorphic_execution.h"
#include "../common/support.h"

namespace grppi {

template <typename Generator, typename Consumer>
void farm_multi_impl(polymorphic_execution & ex, Generator generate_op,
                     Consumer consume_op) 
{
}


template <typename Generator, typename Transformer, typename Consumer>
void farm_multi_impl(polymorphic_execution & ex, Generator generate_op, 
                     Transformer transform_op, Consumer consume_op) 
{
}

template <typename E, typename ... O, typename Generator, typename Transformer,
          internal::requires_execution_not_supported<E> = 0>
void farm_multi_impl(polymorphic_execution & ex, Generator && generate_op, 
                     Transformer && transform_op)
{
  farm_multi_impl<O...>(ex, std::forward<Generator>(generate_op), 
      std::forward<Transformer>(transform_op));
}

template <typename E, typename ... O, typename Generator, typename Transformer, 
          typename Consumer,
          internal::requires_execution_not_supported<E> = 0>
void farm_multi_impl(polymorphic_execution & ex, Generator && generate_op, 
                     Transformer && transform_op, Consumer && consume_op)
{
  farm_multi_impl<O...>(ex, std::forward<Generator>(generate_op),
      std::forward<Transformer>(transform_op), 
      std::forward<Consumer>(consume_op));
}



template <typename E, typename ... O, typename Generator, typename Consumer,
          internal::requires_execution_supported<E> = 0>
void farm_multi_impl(polymorphic_execution & ex, Generator && generate_op,
                     Consumer && consume_op)
{
  if (typeid(E) == ex.type()) {
    farm(*ex.execution_ptr<E>(), std::forward<Generator>(generate_op),
        std::forward<Consumer>(consume_op));
  }
  else {
    farm_multi_impl<O...>(ex, std::forward<Generator>(generate_op), 
        std::forward<Consumer>(consume_op));
  }
}


template <typename E, typename ... O, typename Generator, typename Transformer, typename Consumer,
          internal::requires_execution_supported<E> = 0>
void farm_multi_impl(polymorphic_execution & ex, Generator && generate_op, 
                     Transformer && transform_op, Consumer && consume_op)
{
  if (typeid(E) == ex.type()) {
    farm(*ex.execution_ptr<E>(), std::forward<Generator>(generate_op), 
        std::forward<Transformer>(transform_op),
        std::forward<Consumer>(consume_op));
  }
  else {
    farm_multi_impl<O...>(ex, std::forward<Generator>(generate_op), 
        std::forward<Transformer>(transform_op),
        std::forward<Consumer>(consume_op));
  }
}


/**
\addtogroup farm_pattern
@{
\addtogroup farm_pattern_poly Polymorphic execution farm pattern
\brief Polymorphic execution implementation of the \ref md_farm.
@{
*/

/**
\brief Invoke \ref md_farm on a data stream with polymorphic 
execution with a generator and a consumer.
\tparam Generator Callable type for the generation operation.
\tparam Consumer Callable type for the consume operation.
\param ex Polymorphic execution policy object.
\param generate_op Generator operation.
\param consume_op Consumer operation.
*/
template <typename Generator, typename Consumer>
void farm(polymorphic_execution & ex, Generator && generate_op, 
          Consumer && consume_op)
{
  farm_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb,
	parallel_execution_ff
  >(ex, std::forward<Generator>(generate_op), 
      std::forward<Consumer>(consume_op));
}

/**
\brief Invoke \ref md_farm on a data stream with polymorphic
execution with a generator and a consumer.
\tparam Generator Callable type for the generation operation.
\tparam Tranformer Callable type for the tranformation operation.
\tparam Consumer Callable type for the consume operation.
\param ex Polymorphic execution policy object.
\param generate_op Generator operation.
\param transform_op Transformer operation.
\param consume_op Consumer operation.
*/
template <typename Generator, typename Transformer, typename Consumer>
void farm(polymorphic_execution & ex, Generator && generate_op, 
          Transformer && transform_op, Consumer && consume_op)
{
  farm_multi_impl<
    sequential_execution,
    parallel_execution_native,
    parallel_execution_omp,
    parallel_execution_tbb,
	parallel_execution_ff
  >(ex, std::forward<Generator>(generate_op), 
      std::forward<Transformer>(transform_op), 
      std::forward<Consumer>(consume_op));
}

} // end namespace grppi

#endif
